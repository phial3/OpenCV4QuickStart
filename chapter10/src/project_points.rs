use anyhow::Result;
use opencv::{
    calib3d,
    core::{Mat, Point2f, Point3f, Size, Vector},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter10/";

pub(crate) fn run() -> Result<()> {
    // 输入相机内参矩阵和畸变系数（已知标定结果）
    let camera_matrix = Mat::from_slice_2d(&[
        &[532.016297, 0.0, 332.172519],
        &[0.0, 531.565159, 233.388075],
        &[0.0, 0.0, 1.0],
    ])?;

    let dist_coeffs = Mat::from_slice(&[-0.285188, 0.080097, 0.001274, -0.002415, 0.106579])?.try_clone()?;

    // 旋转向量和平移向量（从标定中得到的相机坐标系与世界坐标系之间的关系）
    let rvec = Mat::from_slice(&[-1.977853, -2.002220, 0.130029])?.try_clone()?;
    let tvec = Mat::from_slice(&[-26.88155, -42.79936, 159.19703])?.try_clone()?;

    // 生成棋盘格内角点的三维世界坐标
    let board_size = Size::new(9, 6);
    let square_size = Size::new(10, 10); // 棋盘格每个方格的实际尺寸
    let mut point_sets = Vector::<Point3f>::new();
    for j in 0..board_size.height {
        for k in 0..board_size.width {
            let real_point = Point3f::new(
                (j * square_size.width) as f32,
                (k * square_size.height) as f32,
                0.0, // 假设标定板为世界坐标系的z=0平面
            );
            point_sets.push(real_point);
        }
    }

    // 根据三维坐标和相机与世界坐标系之间的关系计算内角点像素坐标
    let mut image_points = Vector::<Point2f>::new();
    calib3d::project_points(&point_sets, &rvec, &tvec, &camera_matrix, &dist_coeffs,
                            &mut image_points, &mut opencv::core::no_array(), 0.0)?;

    /********** 计算图像中的内角点的真实坐标误差 **********/
    // 读取图像
    let img = imgcodecs::imread(&(BASE_PATH.to_string() + "left01.jpg"), imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut img_points = Vector::<Point2f>::new();
    calib3d::find_chessboard_corners(&gray, board_size, &mut img_points, calib3d::CALIB_CB_ADAPTIVE_THRESH)?;
    calib3d::find4_quad_corner_subpix(&gray, &mut img_points, Size::new(5, 5))?;

    // 计算估计值和图像中计算的真实值之间的平均误差
    let mut e = 0.0;
    for i in 0..image_points.len() {
        let e_x = (image_points.get(i)?.x - img_points.get(i)?.x).powf(2.0);
        let e_y = (image_points.get(i)?.y - img_points.get(i)?.y).powf(2.0);
        e += (e_x + e_y).sqrt();
    }
    e /= image_points.len() as f32;

    // 输出误差
    println!("估计坐标与真实坐标之间的误差: {}", e);

    // 等待按键
    highgui::wait_key(0)?;

    Ok(())
}
