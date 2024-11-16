
use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, Vec2f, Point2f, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    let img = imgcodecs::imread(&(BASE_PATH.to_owned() + "HoughLines.jpg"), imgcodecs::IMREAD_GRAYSCALE)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    let mut edge = Mat::default();
    // 检测边缘图像，并二值化
    imgproc::canny(&img, &mut edge, 80.0, 180.0, 3, false)?;
    imgproc::threshold(&edge.clone(), &mut edge, 170.0, 255.0, imgproc::THRESH_BINARY)?;

    // 用不同的累加器进行检测直线
    let mut lines1 = Vector::<Vec2f>::new();
    let mut lines2 = Vector::<Vec2f>::new();
    imgproc::hough_lines(&edge, &mut lines1, 1.0, opencv::core::CV_PI / 180.0, 50, 0.0, 0.0, 0.0, opencv::core::CV_PI)?;
    imgproc::hough_lines(&edge, &mut lines2, 1.0, opencv::core::CV_PI / 180.0, 150, 0.0, 0.0,0.0, opencv::core::CV_PI)?;

    // 在原图像中绘制直线
    let mut img1 = img.clone();
    let mut img2 = img.clone();
    draw_line(&mut img1, &lines1, edge.rows() as f64, edge.cols() as f64, Scalar::new(255.0, 255.0, 255.0, 0.0), 2);
    draw_line(&mut img2, &lines2, edge.rows() as f64, edge.cols() as f64, Scalar::new(255.0, 255.0, 255.0, 0.0), 2);

    // 显示图像
    highgui::imshow("edge", &edge)?;
    highgui::imshow("img", &img)?;
    highgui::imshow("img1", &img1)?;
    highgui::imshow("img2", &img2)?;

    highgui::wait_key(0)?;

    Ok(())
}

fn draw_line(img: &mut Mat, lines: &Vector<Vec2f>, rows: f64, cols: f64, color: Scalar, thickness: i32) {
    let mut pt1 = Point::default();
    let mut pt2 = Point::default();
    for i in 0..lines.len() {
        let line = lines.get(i).unwrap();
        let rho = line[0]; // 直线距离坐标原点的距离
        let theta = line[1]; // 直线过坐标原点垂线与x轴夹角
        let a = theta.cos(); // 夹角的余弦值
        let b = theta.sin(); // 夹角的正弦值
        let x0 = a * rho;
        let y0 = b * rho; // 直线与过坐标原点的垂线的交点
        let length = rows.max(cols) as f32; // 图像高宽的最大值

        // 计算直线上的一点
        pt1.x = (x0 + length * -b).round() as i32;
        pt1.y = (y0 + length * a).round() as i32;

        // 计算直线上另一点
        pt2.x = (x0 - length * -b).round() as i32;
        pt2.y = (y0 - length * a).round() as i32;

        // 两点绘制一条直线
        imgproc::line(img, pt1, pt2, color, thickness, 8, 0).unwrap();
    }
}