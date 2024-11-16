

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
    let mut img = imgcodecs::imread(&(BASE_PATH.to_owned() + "stuff.jpg"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 深拷贝
    let mut img1 = img.clone();  // 用于绘制最大外接矩形
    let mut img2 = img.clone();  // 用于绘制最小外接矩形
    let mut img3 = img.clone();  // 用于绘制中心点
    // 显示原始图像
    highgui::imshow("img", &img)?;

    // 去噪声与二值化
    let mut canny = Mat::default();
    imgproc::canny(&img, &mut canny, 80.0, 160.0, 3, false)?;
    highgui::imshow("Canny", &canny)?;

    // 膨胀运算，将细小缝隙填补上
    let kernel = imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(3, 3), Point::new(-1, -1))?;
    imgproc::dilate(&canny.clone(), &mut canny, &kernel, Point::new(-1, -1), 1, opencv::core::BORDER_CONSTANT, Scalar::default())?;

    // 轮廓发现与绘制
    let mut contours = Vector::<Mat>::new();
    let mut hierarchy = Vector::<Vec4i>::new();
    imgproc::find_contours_with_hierarchy(&canny,
                                          &mut contours,
                                          &mut hierarchy,
                                          imgproc::RETR_EXTERNAL,
                                          imgproc::CHAIN_APPROX_SIMPLE,
                                          Point::new(0, 0))?;

    // 寻找轮廓的外接矩形
    for n in 0..contours.len() {
        // 最大外接矩形
        let rect = imgproc::bounding_rect(&contours.get(n)?)?;
        imgproc::rectangle(&mut img1, rect, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;

        // 最小外接矩形
        let rrect = imgproc::min_area_rect(&contours.get(n)?)?;
        let mut points: [Point2f; 4] = Default::default();
        rrect.points(&mut points)?;
        let cpt = rrect.center;  // 最小外接矩形的中心

        // 绘制旋转矩形与中心位置
        for i in 0..4 {
            let start = points[i];
            let end = points[(i + 1) % 4];
            imgproc::line(&mut img2,
                          Point::new(start.x as i32, start.y as i32),
                          Point::new(end.x as i32, end.y as i32),
                          Scalar::new(0.0, 255.0, 0.0, 0.0),
                          2, imgproc::LINE_8, 0)?;
        }

        // 绘制矩形的中心
        imgproc::circle(&mut img3,
                        Point::new(cpt.x as i32, cpt.y as i32),
                        2,
                        Scalar::new(255.0, 0.0, 0.0, 0.0),
                        2,
                        imgproc::LINE_8,
                        0)?;
    }

    // 输出绘制外接矩形的结果
    highgui::imshow("max", &img1)?;
    highgui::imshow("min", &img2)?;
    highgui::imshow("centers", &img3)?;
    highgui::wait_key(0)?;

    Ok(())
}
