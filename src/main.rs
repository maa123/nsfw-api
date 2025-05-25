use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Instant;

use hyper::body::{Body as _, Incoming}; // For Body::collect
use hyper::{Request, Response};
use image::io::Reader;
use tract_onnx::prelude::*;
use tokio::net::TcpListener;
use hyper_util::rt::{TokioIo, TokioExecutor};
use http_body_util::{BodyExt, Full, Empty}; // For BodyExt::collect, Full, Empty
use bytes::Bytes; // Required for Full<Bytes>
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use std::sync::Arc; // For Arc<Model>
use hyper::service::service_fn; // service_fn is now directly in hyper::service
use serde_derive::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct NConfig {
    address: String,
}

impl ::std::default::Default for NConfig {
    fn default() -> Self { Self { address: "127.0.0.1:3000".into() } }
}

pub trait VecResultArray {
    fn to_result_array(self) -> [f32; 5];
}

impl VecResultArray for Vec<f32> {
    fn to_result_array(self) -> [f32; 5] {
        self.try_into().unwrap()
    }
}

struct Struc<T>(Arc<T>); // Wrap model in Arc for sharing

struct CheckResult {
    result: [f32; 5],
    time: u128,
}

async fn fetch(url: &str) -> Result<std::io::Cursor<Vec<u8>>, reqwest::Error> {
    let body = reqwest::get(url).await?.bytes().await?;
    Ok(std::io::Cursor::new(body.to_vec()))
}

async fn get_body(body: Incoming) -> Result<std::io::Cursor<Vec<u8>>, hyper::Error> {
    // Use BodyExt::collect to get the full body
    let collected_body = body.collect().await?;
    // Convert collected body to bytes
    let bytes = collected_body.to_bytes();
    Ok(std::io::Cursor::new(bytes.to_vec()))
}

fn load_model() -> Result<Arc<Graph<TypedFact, Box<dyn TypedOp>>>, ()> { // Return Arc<Model>
    let model = tract_onnx::onnx()
        .model_for_path("model.onnx")
        .unwrap()
        .into_optimized()
        .unwrap();
    Ok(Arc::new(model)) // Wrap model in Arc
}

async fn run(
    img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    model: Arc<Graph<TypedFact, Box<dyn TypedOp>>>, // Expect Arc<Model>
) -> Result<CheckResult, ()> {
    // Clone the Arc for use in this async task, if needed, or pass by value if plan can take Arc
    // For SimplePlan::new, it might take ownership or a reference.
    // If it takes ownership and we need the model elsewhere, model.as_ref() or clone the Arc.
    // Assuming SimplePlan::new can take the model by value (moving the Arc) or by reference (model.as_ref())
    let plan = SimplePlan::new(model.as_ref().clone()).unwrap(); // Use as_ref() if new() expects a reference, or clone the inner model if necessary
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
    req: Request<Incoming>, // Updated Request type
    model_arc: Struc<Graph<TypedFact, Box<dyn TypedOp>>>, // Expecting Struc(Arc<Model>)
) -> Result<Response<Full<Bytes>>, Infallible> { // Updated Response type
    let model = model_arc.0; // model is Arc<Graph<...>>
    match req.method() {
        &hyper::Method::GET => {
            if let Some(img_url) =
                url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
                    .find(|(k, _)| k == "url")
                    .map(|(_, v)| v.to_string())
            {
                let img_fetch_result = fetch(img_url.as_str()).await;
                if let Ok(img_cursor) = img_fetch_result {
                    let img_reader_result = Reader::new(img_cursor)
                        .with_guessed_format();
                    if let Ok(img_reader) = img_reader_result {
                        let decode_result = img_reader.decode();
                        if let Ok(decoded_img) = decode_result {
                            let rgb_img = decoded_img.to_rgb8();
                            // Clone the Arc for the run function
                            let res = run(rgb_img, model.clone()).await;
                            if let Ok(res_check) = res {
                                let response_body = Full::from(Bytes::from(format!("{:?}", res_check.result)));
                                let response = Response::builder()
                                    .status(200)
                                    .header("Content-Type", "application/json")
                                    .header("NC-time", res_check.time.to_string())
                                    .body(response_body)
                                    .unwrap();
                                return Ok(response);
                            }
                        }
                    }
                }
                let response_body = Full::from(Bytes::from_static(b"Internal Server Error"));
                let internal_server_error = Response::builder()
                    .status(500)
                    .body(response_body)
                    .unwrap();
                return Ok(internal_server_error);
            }
            let response_body = Full::from(Bytes::from_static(b"Bad Request"));
            let bad_request = Response::builder()
                .status(400)
                .body(response_body)
                .unwrap();
            Ok(bad_request)
        }
        &hyper::Method::POST => {
            // req.into_body() is already Incoming
            let body_cursor_result = get_body(req.into_body()).await;
            if let Ok(body_cursor) = body_cursor_result {
                 let img_reader_result = Reader::new(body_cursor)
                    .with_guessed_format();
                if let Ok(img_reader) = img_reader_result {
                    let decode_result = img_reader.decode();
                    if let Ok(decoded_img) = decode_result {
                        let rgb_img = decoded_img.to_rgb8();
                        // Clone the Arc for the run function
                        let res = run(rgb_img, model.clone()).await;
                        if let Ok(res_check) = res {
                            let response_body = Full::from(Bytes::from(format!("{:?}", res_check.result)));
                            let response = Response::builder()
                                .status(200)
                                .header("Content-Type", "application/json")
                                .header("NC-time", res_check.time.to_string())
                                .body(response_body)
                                .unwrap();
                            return Ok(response);
                        }
                    }
                }
            }
            let response_body = Full::from(Bytes::from_static(b"Internal Server Error"));
            let internal_server_error = Response::builder()
                .status(500)
                .body(response_body)
                .unwrap();
            Ok(internal_server_error)
        }
        _ => {
            let response_body = Full::from(Bytes::from_static(b"Not Found"));
            let not_found = Response::builder()
                .status(404)
                .body(response_body)
                .unwrap();
            Ok(not_found)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config: NConfig = confy::load_path("./config.toml").unwrap();
    let model_arc = load_model().unwrap(); // model_arc is Arc<Graph<...>>
    let addr: SocketAddr = config.address.parse().unwrap();

    let listener = TcpListener::bind(addr).await?;
    println!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        
        // Clone the Arc for the service_fn closure
        let model_clone_for_service = model_arc.clone(); 

        let service = service_fn(move |req: Request<Incoming>| {
            // Further clone the Arc for the handle function call
            let model_clone_for_handle = model_clone_for_service.clone();
            handle(req, Struc(model_clone_for_handle))
        });

        // Spawn a tokio task to serve multiple connections concurrently
        tokio::task::spawn(async move {
            if let Err(err) = AutoBuilder::new(TokioExecutor::new())
                .serve_connection(io, service)
                .await
            {
                eprintln!("Error serving connection: {}", err);
            }
        });
    }
}
