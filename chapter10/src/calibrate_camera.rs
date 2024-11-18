use anyhow::{Result, Error, Context};
use std::{fs::File, io::{BufRead, BufReader}};
use opencv::{
    prelude::*,
    core::{Mat, Point2f, Point3f, Scalar, Size, Vector, TermCriteria, TermCriteria_Type},
    imgcodecs,
    imgproc,
    highgui,
    calib3d,
    features2d,
};

const BASE_PATH: &str = "../data/chapter10/";

pub(crate) fn run() -> Result<()> {
    // 读取所有图像
    let mut imgs = Vec::new();
    let file = File::open(format!("{}calibdata.txt", BASE_PATH))?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let image_name = line?;
        let img = imgcodecs::imread(
            &format!("{}{}", BASE_PATH, image_name),
            imgcodecs::IMREAD_COLOR,
        )?;
        imgs.push(img);
    }

    // 方格标定板内角点数目（行，列）
    let board_size = Size::new(9, 6);

    // 多个图的定点坐标集
    let mut imgs_points = Vector::<Vector::<Point2f>>::new();

    // 处理每张图像
    for img in &imgs {
        let mut gray = Mat::default();
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut img_points = Vector::<Point2f>::new();
        calib3d::find_chessboard_corners(
            &gray,
            board_size,
            &mut img_points,
            calib3d::CALIB_CB_ADAPTIVE_THRESH,
        )?;

        // 细化方格标定板角点坐标
        let criteria = TermCriteria::new(
            opencv::core::TermCriteria_COUNT + opencv::core::TermCriteria_EPS,
            30,
            0.1,
        )?;

        imgproc::corner_sub_pix(
            &gray,
            &mut img_points,
            Size::new(5, 5),
            Size::new(-1, -1),
            criteria,
        )?;

        imgs_points.push(img_points);
    }

    // 生成棋盘格每个内角点的空间三维坐标
    let square_size = Size::new(10, 10);  // 棋盘格每个方格的真实尺寸
    let mut object_points = Vector::<Vector::<Point3f>>::new();

    for _ in 0..imgs_points.len() {
        let mut temp_point_set = Vector::<Point3f>::new();
        for j in 0..board_size.height {
            for k in 0..board_size.width {
                let real_point = Point3f::new(
                    j as f32 * square_size.width as f32,
                    k as f32 * square_size.height as f32,
                    0.0,
                );
                temp_point_set.push(real_point);
            }
        }
        object_points.push(temp_point_set);
    }

    // 初始化每幅图像中的角点数量
    let point_numbers: Vec<i32> = (0..imgs_points.len())
        .map(|_| board_size.width * board_size.height)
        .collect();
    println!("角点数量={:?}", point_numbers);

    // 图像尺寸
    let image_size = Size::new(imgs[0].cols(), imgs[0].rows());

    // 摄像机内参数矩阵
    let mut camera_matrix = Mat::new_rows_cols_with_default(
        3,
        3,
        opencv::core::CV_32FC1,
        Scalar::all(0.0),
    )?;

    // 摄像机的5个畸变系数：k1,k2,p1,p2,k3
    let mut dist_coeffs = Mat::new_rows_cols_with_default(
        1,
        5,
        opencv::core::CV_32FC1,
        Scalar::all(0.0),
    )?;

    let mut rvecs = Vector::<Mat>::new();  // 每幅图像的旋转向量
    let mut tvecs = Vector::<Mat>::new();  // 每张图像的平移向量

    // 相机标定
    calib3d::calibrate_camera(
        &object_points,
        &imgs_points,
        image_size,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        0,
        TermCriteria::new(opencv::core::TermCriteria_COUNT + opencv::core::TermCriteria_EPS,30, 0.001)?
    )?;

    println!("相机的内参矩阵=\n{:?}", camera_matrix);
    println!("相机畸变系数={:?}", dist_coeffs);

    highgui::wait_key(0)?;

    Ok(())
}
