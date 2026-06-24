use std::convert::Infallible;
use std::io::Cursor;
use std::net::SocketAddr;
use std::time::Instant;

use anyhow::{Context, Result};
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response};
use hyper_util::rt::TokioIo;
use image::ImageReader;
use serde_derive::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tract_onnx::prelude::*;

#[derive(Serialize, Deserialize)]
struct NConfig {
    address: String,
    add_nc_time_header: bool,
}

impl ::std::default::Default for NConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:3000".into(),
            add_nc_time_header: true,
        }
    }
}

pub trait VecResultArray {
    fn to_result_array(self) -> Result<[f32; 5]>;
}

impl VecResultArray for Vec<f32> {
    fn to_result_array(self) -> Result<[f32; 5]> {
        let len = self.len();
        self.try_into()
            .map_err(|_| anyhow::anyhow!("Expected 5 model outputs, got {len}"))
    }
}

struct Struc<T>(T);

struct CheckResult {
    result: [f32; 5],
    time: u128,
}

type RespBody = Full<Bytes>;

fn body_from<T: Into<Bytes>>(body: T) -> RespBody {
    Full::new(body.into())
}

async fn fetch(url: &str) -> Result<Cursor<Vec<u8>>, reqwest::Error> {
    let body = reqwest::get(url).await?.bytes().await?;
    Ok(Cursor::new(body.to_vec()))
}

async fn get_body(body: Incoming) -> Result<Cursor<Vec<u8>>, hyper::Error> {
    let body = body.collect().await?.to_bytes();
    Ok(Cursor::new(body.to_vec()))
}

fn load_model() -> Result<Graph<TypedFact, Box<dyn TypedOp>>> {
    let model = tract_onnx::onnx()
        .model_for_path("model.onnx")
        .context("Failed to load model from model.onnx")?
        .into_optimized()
        .context("Failed to optimize model")?;
    Ok(model)
}

async fn run(
    img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    model: Graph<TypedFact, Box<dyn TypedOp>>,
) -> Result<CheckResult> {
    let plan = SimplePlan::new(model).context("Failed to create inference plan")?;
    let start = Instant::now();
    let img = image::imageops::resize(&img, 224, 224, image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        (img[(x as _, y as _)][c] as f32) / 255.0
    })
    .into();
    let result = plan
        .run(tvec!(image.into()))
        .context("Failed to run model inference")?;
    let result: Vec<f32> = result[0]
        .to_array_view::<f32>()
        .context("Failed to read inference result")?
        .iter()
        .copied()
        .collect();
    Ok(CheckResult {
        result: result.to_result_array()?,
        time: start.elapsed().as_millis(),
    })
}

fn internal_server_error() -> Response<RespBody> {
    Response::builder()
        .status(500)
        .body(body_from("Internal Server Error"))
        .expect("Failed to build 500 response")
}

async fn handle(
    req: Request<Incoming>,
    model: Struc<Graph<TypedFact, Box<dyn TypedOp>>>,
    add_nc_time_header: bool,
) -> Result<Response<RespBody>, Infallible> {
    let model = model.0;
    match req.method() {
        &Method::GET => {
            if let Some(img_url) =
                url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
                    .find(|(k, _)| k == "url")
                    .map(|(_, v)| v.to_string())
            {
                let result: Result<Response<RespBody>> = async {
                    let img = fetch(img_url.as_str()).await.context("Failed to fetch image")?;
                    let img = ImageReader::new(img)
                        .with_guessed_format()
                        .context("Failed to guess image format")?
                        .decode()
                        .context("Failed to decode image")?
                        .to_rgb8();
                    let res = run(img, model).await?;
                    let mut response_builder = Response::builder();
                    response_builder = response_builder.status(200);
                    response_builder = response_builder.header("Content-Type", "application/json");
                    if add_nc_time_header {
                        response_builder = response_builder.header("NC-time", res.time.to_string());
                    }
                    let response = response_builder
                        .body(body_from(format!("{:?}", res.result)))
                        .expect("Failed to build 200 response");
                    Ok(response)
                }
                .await;

                return Ok(result.unwrap_or_else(|err| {
                    eprintln!("Error handling GET request: {err:#}");
                    internal_server_error()
                }));
            }
            let bad_request = Response::builder()
                .status(400)
                .body(body_from("Bad Request"))
                .expect("Failed to build 400 response");
            Ok(bad_request)
        }
        &Method::POST => {
            let result: Result<Response<RespBody>> = async {
                let body = get_body(req.into_body())
                    .await
                    .context("Failed to read request body")?;
                let img = ImageReader::new(body)
                    .with_guessed_format()
                    .context("Failed to guess image format")?
                    .decode()
                    .context("Failed to decode image")?
                    .to_rgb8();
                let res = run(img, model).await?;
                let mut response_builder = Response::builder();
                response_builder = response_builder.status(200);
                response_builder = response_builder.header("Content-Type", "application/json");
                if add_nc_time_header {
                    response_builder = response_builder.header("NC-time", res.time.to_string());
                }
                let response = response_builder
                    .body(body_from(format!("{:?}", res.result)))
                    .expect("Failed to build 200 response");
                Ok(response)
            }
            .await;

            Ok(result.unwrap_or_else(|err| {
                eprintln!("Error handling POST request: {err:#}");
                internal_server_error()
            }))
        }
        _ => {
            let not_found = Response::builder()
                .status(404)
                .body(body_from("Not Found"))
                .expect("Failed to build 404 response");
            Ok(not_found)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config: NConfig = confy::load_path("./config.toml")?;
    let model = load_model()?;
    let addr: SocketAddr = config.address.parse()?;
    let add_nc_time_header = config.add_nc_time_header;
    let listener = TcpListener::bind(addr).await?;
    println!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let model = model.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |req| {
                let model_struc = Struc(model.clone());
                handle(req, model_struc, add_nc_time_header)
            });

            if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                eprintln!("server connection error: {err}");
            }
        });
    }
}
