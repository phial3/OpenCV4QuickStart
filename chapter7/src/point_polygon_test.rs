
use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, Point2f, Point2d, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {

    // 读取图像
    let mut img = imgcodecs::imread(&(BASE_PATH.to_owned() + "approx.png"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 边缘检测
    let mut canny = Mat::default();
    imgproc::canny(
        &img,
        &mut canny,
        80.0,
        160.0,
        3,
        false
    )?;
    // 膨胀运算
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 3),
        Point::new(-1, -1)
    )?;

    imgproc::dilate(
        &canny.clone(),
        &mut canny,
        &kernel,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::default()
    )?;

    // 轮廓发现
    let mut contours = Vector::<Mat>::new();
    let mut hierarchy = Vector::<Vec4i>::new();

    imgproc::find_contours_with_hierarchy(
        &canny,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;
    // 创建图像中的一个像素点并绘制圆形
    let point = Point::new(250, 200);
    imgproc::circle(
        &mut img,
        point,
        2,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        0
    )?;
    // 多边形处理
    for t in 0..contours.len() {
        let contour = contours.get(t)?;
        // 用最小外接矩形求取轮廓中心
        let rrect = imgproc::min_area_rect(&contour)?;
        let center = rrect.center;
        // 绘制圆心点
        imgproc::circle(
            &mut img,
            Point::new(center.x as i32, center.y as i32),
            2,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0
        )?;
        // 轮廓外部点距离轮廓的距离
        let dis = imgproc::point_polygon_test(
            &contour,
            Point2f::new(point.x as f32, point.y as f32),
            true
        )?;
        // 轮廓内部点距离轮廓的距离
        let dis2 = imgproc::point_polygon_test(
            &contour,
            center,
            true
        )?;
        // 输出点结果
        println!("外部点距离轮廓距离：{}", dis);
        println!("内部点距离轮廓距离：{}", dis2);
    }

    // 可选：保存结果图像
    imgcodecs::imwrite(&(BASE_PATH.to_owned() + "result.png"), &img, &Vector::default())?;

    Ok(())
}