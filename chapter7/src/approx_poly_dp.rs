use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec2i, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    let img = imgcodecs::imread(&format!("{}approx.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 边缘检测
    let mut canny = Mat::default();
    imgproc::canny(&img, &mut canny, 80.0, 160.0, 3, false)?;

    // 膨胀运算
    let kernel = imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(3, 3), Point::new(-1, -1))?;
    imgproc::dilate(&canny.clone(), &mut canny, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::all(0.0))?;

    // 轮廓发现与绘制
    let mut contours = Vector::<Vector::<Point>>::new();
    imgproc::find_contours(&canny, &mut contours, imgproc::RETR_CCOMP, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    // 绘制多边形
    for contour in contours.iter() {
        let rrect = imgproc::min_area_rect(&contour)?;
        // 将 Point2f 转换为 Point
        let center = Point::new(rrect.center.x as i32, rrect.center.y as i32);
        imgproc::circle(&mut img.clone(), center, 2, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, 8, 0)?;

        let mut result = Mat::default();
        imgproc::approx_poly_dp(&contour, &mut result, 4.0, true)?;

        drawapp(&result, &mut img.clone())?;
        println!("corners : {}", result.rows());

        // 判断形状和绘制轮廓
        match result.rows() {
            3 => imgproc::put_text(&mut img.clone(), "triangle", center, imgproc::FONT_HERSHEY_SIMPLEX, 1.0, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, 8, false)?,
            4 => imgproc::put_text(&mut img.clone(), "rectangle", center, imgproc::FONT_HERSHEY_SIMPLEX, 1.0, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, 8, false)?,
            8 => imgproc::put_text(&mut img.clone(), "poly-8", center, imgproc::FONT_HERSHEY_SIMPLEX, 1.0, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, 8, false)?,
            n if n > 12 => imgproc::put_text(&mut img.clone(), "circle", center, imgproc::FONT_HERSHEY_SIMPLEX, 1.0, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, 8, false)?,
            _ => {}
        }
    }
    highgui::imshow("result", &img)?;
    highgui::wait_key(0)?;

    Ok(())
}

fn drawapp(result: &Mat, img: &mut Mat) -> Result<()> {
    let rows = result.rows();
    for i in 0..rows {
        let point1 = result.at::<Vec2i>(i)?;
        let point2 = if i == rows - 1 {
            result.at::<Vec2i>(0)?
        } else {
            result.at::<Vec2i>(i + 1)?
        };

        // 将 Vec2i 转换为 Point
        let pt1 = Point::new(point1[0], point1[1]);
        let pt2 = Point::new(point2[0], point2[1]);
        imgproc::line(img, pt1, pt2, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, 8, 0)?;
    }
    Ok(())
}