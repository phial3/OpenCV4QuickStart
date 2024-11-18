use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use opencv::{
    calib3d,
    core::{Mat, Point2f, Point3f, Size, Vector, TermCriteria},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter10/";

pub(crate) fn run() -> Result<()> {
    // 读取所有图像
    let mut imgLs= Vector::<Mat>::new();
    let mut imgRs = Vector::<Mat>::new();

    let file_l = File::open(BASE_PATH.to_string() + "steroCalibDataL.txt")?;
    let file_r = File::open(BASE_PATH.to_string() + "steroCalibDataR.txt")?;

    let mut finL = BufReader::new(file_l);
    let mut finR = BufReader::new(file_r);

    let mut imgLName = String::new();
    let mut imgRName = String::new();

    while finL.read_line(&mut imgLName)? > 0 && finR.read_line(&mut imgRName)? > 0 {
        imgLName = imgLName.trim().to_string();
        imgRName = imgRName.trim().to_string();

        println!("image read imgLName={}, imgRName={}", imgLName, imgRName);
        let imgL = imgcodecs::imread(&(BASE_PATH.to_string() + &imgLName), imgcodecs::IMREAD_COLOR)?;
        let imgR = imgcodecs::imread(&(BASE_PATH.to_string() + &imgRName), imgcodecs::IMREAD_COLOR)?;
        if imgL.empty() || imgR.empty() {
            panic!("图像读取失败，请检查文件路径");
        }

        imgLs.push(imgL);
        imgRs.push(imgR);
    }

    // 提取棋盘格内角点在两个相机图像中的坐标
    let board_size = Size::new(9, 6); // 方格标定板内角点数目（行，列）
    let imgLsPoints = get_imgs_points(imgLs.clone(), board_size)?;
    let imgRsPoints = get_imgs_points(imgRs, board_size)?;

    // 生成棋盘格每个内角点的空间三维坐标
    let square_size = Size::new(10, 10); // 棋盘格每个方格的真实尺寸
    let mut object_points= Vector::<Vector<Point3f>>::new();

    for _ in 0..imgLsPoints.len() {
        let mut temp_point_set = Vector::<Point3f>::new();
        for j in 0..board_size.height {
            for k in 0..board_size.width {
                let real_point = Point3f::new(
                    (j * square_size.width) as f32,
                    (k * square_size.height) as f32,
                    0.0, // 假设标定板为世界坐标系的z=0平面
                );
                temp_point_set.push(real_point);
            }
        }
        object_points.push(temp_point_set);
    }

    // 图像尺寸
    let i0 = imgLs.get(0)?;
    let image_size = Size::new(i0.cols(), i0.rows());

    // 校准左相机和右相机
    let mut matrix1 = Mat::default();
    let mut dist1 = Mat::default();
    let mut matrix2 = Mat::default();
    let mut dist2 = Mat::default();
    let mut rvecs = Mat::default();
    let mut tvecs = Mat::default();

    calib3d::calibrate_camera(&object_points, &imgLsPoints, image_size, &mut matrix1, &mut dist1, &mut rvecs, &mut tvecs, 0, TermCriteria::default().unwrap())?;
    calib3d::calibrate_camera(&object_points, &imgRsPoints, image_size, &mut matrix2, &mut dist2, &mut rvecs, &mut tvecs, 0, TermCriteria::default().unwrap())?;

    // 进行立体标定
    let mut R = Mat::default();
    let mut T = Mat::default();
    let mut E = Mat::default();
    let mut F = Mat::default();

    calib3d::stereo_calibrate(
        &object_points,
        &imgLsPoints,
        &imgRsPoints,
        &mut matrix1,
        &mut dist1,
        &mut matrix2,
        &mut dist2,
        image_size,
        &mut R,
        &mut T,
        &mut E,
        &mut F,
        calib3d::CALIB_USE_INTRINSIC_GUESS,
        TermCriteria::default().unwrap()
    )?;

    // 输出旋转矩阵和平移向量
    println!("两个相机坐标系的旋转矩阵：{:?}", R);
    println!("两个相机坐标系的平移向量：{:?}", T);

    highgui::wait_key(0)?;

    Ok(())
}

fn get_imgs_points(imgs: Vector<Mat>, board_size: Size) -> opencv::Result<Vector<Vector<Point2f>>> {
    let mut points= Vector::<Vector<Point2f>>::new();
    for img in imgs {
        let mut gray1 = Mat::default();
        imgproc::cvt_color(&img, &mut gray1, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut img1_points= Vector::<Point2f>::new();
        calib3d::find_chessboard_corners(&gray1, board_size, &mut img1_points, calib3d::CALIB_CB_ADAPTIVE_THRESH)?;
        calib3d::find4_quad_corner_subpix(&gray1, &mut img1_points, Size::new(5, 5))?;

        points.push(img1_points);
    }
    Ok(points)
}

