use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Instant;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use image::io::Reader;
use tract_onnx::prelude::*;

pub trait VecResultArray {
    fn to_result_array(self) -> [f32; 5];
}

impl VecResultArray for Vec<f32> {
    fn to_result_array(self) -> [f32; 5] {
        self.try_into().unwrap()
    }
}

struct Struc<T>(T);

struct CheckResult {
    result: [f32; 5],
    time: u128,
}

async fn fetch(url: &str) -> Result<std::io::Cursor<Vec<u8>>, reqwest::Error> {
    let body = reqwest::get(url).await?.bytes().await?;
    Ok(std::io::Cursor::new(body.to_vec()))
}

async fn get_body(body: Body) -> Result<std::io::Cursor<Vec<u8>>, hyper::Error> {
    let body = hyper::body::to_bytes(body).await?;
    Ok(std::io::Cursor::new(body.to_vec()))
}

fn load_model() -> Result<Graph<TypedFact, Box<dyn TypedOp>>, ()> {
    let model = tract_onnx::onnx()
        .model_for_path("model.onnx")
        .unwrap()
        .into_optimized()
        .unwrap();
    Ok(model)
}

async fn run(
    img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    model: Graph<TypedFact, Box<dyn TypedOp>>,
) -> Result<CheckResult, ()> {
    let plan = SimplePlan::new(model).unwrap();
    let start = Instant::now();
    let img = image::imageops::resize(&img, 224, 224, image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        (img[(x as _, y as _)][c] as f32) / 255.0
    })
    .into();
    let result = plan.run(tvec!(image.into())).unwrap();
    let result: Vec<f32> = result[0].to_array_view::<f32>().unwrap().iter().map(|v| *v).collect();
    Ok(CheckResult {
        result: result.to_result_array(),
        time: start.elapsed().as_millis(),
    })
}

async fn handle(
    req: Request<Body>,
    model: Struc<Graph<TypedFact, Box<dyn TypedOp>>>,
) -> Result<Response<Body>, Infallible> {
    let model = model.0;
    match req.method() {
        &hyper::Method::GET => {
            if let Some(img_url) =
                url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
                    .find(|(k, _)| k == "url")
                    .map(|(_, v)| v.to_string())
            {
                let img = fetch(img_url.as_str()).await;
                if let Ok(img) = img {
                    let img = Reader::new(img)
                        .with_guessed_format()
                        .unwrap()
                        .decode()
                        .unwrap()
                        .to_rgb8();
                    let res = run(img, model).await;
                    if let Ok(res) = res {
                        let response = Response::builder()
                            .status(200)
                            .header("Content-Type", "application/json")
                            .header("NC-time", res.time.to_string())
                            .body(Body::from(format!("{:?}", res.result)))
                            .unwrap();
                        return Ok(response);
                    }
                }
                let internal_server_error = Response::builder()
                    .status(500)
                    .body(Body::from("Internal Server Error"))
                    .unwrap();
                return Ok(internal_server_error);
            }
            let bad_request = Response::builder()
                .status(400)
                .body(Body::from("Bad Request"))
                .unwrap();
            Ok(bad_request)
        }
        &hyper::Method::POST => {
            let body = get_body(req.into_body()).await.unwrap();
            let img = Reader::new(body)
                .with_guessed_format()
                .unwrap()
                .decode()
                .unwrap()
                .to_rgb8();
            let res = run(img, model).await;
            if let Ok(res) = res {
                let response = Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .header("NC-time", res.time.to_string())
                    .body(Body::from(format!("{:?}", res.result)))
                    .unwrap();
                return Ok(response);
            }
            let internal_server_error = Response::builder()
                .status(500)
                .body(Body::from("Internal Server Error"))
                .unwrap();
            Ok(internal_server_error)
        }
        _ => {
            let not_found = Response::builder()
                .status(404)
                .body(Body::from("Not Found"))
                .unwrap();
            Ok(not_found)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = load_model().unwrap();
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let model_struc = Struc(model);
    let make_svc = make_service_fn(|_conn| {
        let model_struc = Struc(model_struc.0.clone());
        async {
            Ok::<_, Infallible>(service_fn(move |req| {
                let model_struc = Struc(model_struc.0.clone());
                handle(req, model_struc)
            }))
        }
    });
    let server = Server::bind(&addr).serve(make_svc);
    println!("Listening on http://{}", addr);
    server.await?;
    Ok(())
}
