use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Rect, Vector, Vec3b, Vec4i, RNG},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter8/";

pub(crate) fn run() -> Result<()> {
    // 读取并检查图像
    let img = imgcodecs::imread(&format!("{}HoughLines.jpg", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("请确认图像文件名称是否正确");
    }

    // 创建灰度图和mask
    let mut img_gray = Mat::default();
    let mut img_mask = Mat::default();
    let mut mask_watershed = Mat::default();

    // 转换为灰度图
    imgproc::cvt_color(&img, &mut img_gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Canny边缘检测
    imgproc::canny(&img_gray, &mut img_mask, 150.0, 300.0, 3, false)?;

    // 显示边缘图像和原图像
    highgui::imshow("边缘图像", &img_mask)?;
    highgui::imshow("原图像", &img)?;

    // 查找轮廓
    let mut contours = Vector::<Vector<Point>>::new();
    let mut hierarchy = Vector::<Vec4i>::new();
    imgproc::find_contours_with_hierarchy(
        &img_mask,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_CCOMP,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    )?;

    // 创建watershed的mask
    mask_watershed = Mat::zeros(img_mask.rows(), img_mask.cols(), opencv::core::CV_32S)?.to_mat()?;

    // 在mask上绘制轮廓
    for index in 0..contours.len() {
        imgproc::draw_contours(
            &mut mask_watershed,
            &contours,
            index as i32,
            Scalar::all((index + 1) as f64),
            -1,
            8,
            &hierarchy,
            i32::MAX,
            Point::new(0, 0)
        )?;
    }

    // 应用分水岭算法
    imgproc::watershed(&img, &mut mask_watershed)?;

    // 生成随机颜色
    let mut colors = Vec::new();
    let mut rng = RNG::default()?;
    for _ in 0..contours.len() {
        colors.push(Vec3b::from_array([
            rng.uniform(0, 255)? as u8,
            rng.uniform(0, 255)? as u8,
            rng.uniform(0, 255)? as u8
        ]));
    }

    // 创建结果图像
    let mut result_img = Mat::new_rows_cols_with_default(
        img.rows(),
        img.cols(),
        opencv::core::CV_8UC3,
        Scalar::all(0.0)
    )?;

    // 绘制结果
    unsafe {
        for i in 0..img_mask.rows() {
            for j in 0..img_mask.cols() {
                let index = *mask_watershed.at_2d::<i32>(i, j)?;
                let pixel = result_img.at_2d_mut::<Vec3b>(i, j)?;

                if index == -1 {
                    // 区域间的值被置为-1（边界）
                    *pixel = Vec3b::from_array([255, 255, 255]);
                } else if index <= 0 || index as usize > contours.len() {
                    // 没有标记清楚的区域被置为0
                    *pixel = Vec3b::from_array([0, 0, 0]);
                } else {
                    // 把些区域绘制成不同颜色
                    *pixel = colors[(index - 1) as usize];
                }
            }
        }
    }

    // 混合原图和结果图
    let mut blended = Mat::default();
    opencv::core::add_weighted(&result_img, 0.6, &img, 0.4, 0.0, &mut blended, -1)?;
    highgui::imshow("分水岭结果", &blended)?;

    // 为每个区域创建单独的图像
    for n in 1..=contours.len() {
        let mut res_image = Mat::new_rows_cols_with_default(
            img.rows(),
            img.cols(),
            opencv::core::CV_8UC3,
            Scalar::all(0.0)
        )?;

        unsafe {
            for i in 0..img_mask.rows() {
                for j in 0..img_mask.cols() {
                    let index = *mask_watershed.at_2d::<i32>(i, j)?;
                    if index == n as i32 {
                        *res_image.at_2d_mut::<Vec3b>(i, j)? = *img.at_2d::<Vec3b>(i, j)?;
                    }
                }
            }
        }

        highgui::imshow(&n.to_string(), &res_image)?;
    }

    highgui::wait_key(0)?;

    Ok(())
}

