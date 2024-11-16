
use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec3d, Point2f, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    let mut lines = Mat::default();  // 存放检测直线结果的矩阵
    let mut line3d = Vector::<Vec3d>::new();  // 换一种结果存放形式
    let mut points: Vector<Point2f> = Vector::new();  // 待检测是否存在直线的所有点
    // 定义点
    const POINTS: [[f32; 2]; 20] = [
        [0.0, 369.0],
        [10.0, 364.0],
        [20.0, 358.0],
        [30.0, 352.0],
        [40.0, 346.0],
        [50.0, 341.0],
        [60.0, 335.0],
        [70.0, 329.0],
        [80.0, 323.0],
        [90.0, 318.0],
        [100.0, 312.0],
        [110.0, 306.0],
        [120.0, 300.0],
        [130.0, 295.0],
        [140.0, 289.0],
        [150.0, 284.0],
        [160.0, 277.0],
        [170.0, 271.0],
        [180.0, 266.0],
        [190.0, 260.0],
    ];
    // 将所有点存放在 Vec 中
    for point in POINTS.iter() {
        points.push(Point2f::new(point[0], point[1]));
    }
    // 参数设置
    let rho_min = 0.0;  // 最小长度
    let rho_max = 360.0;  // 最大长度
    let rho_step = 1.0;  // 离散化单位距离长度
    let theta_min = 0.0;  // 最小角度
    let theta_max = std::f64::consts::PI / 2.0;  // 最大角度
    let theta_step = std::f64::consts::PI / 180.0;  // 离散化单位角度弧度

    // 调用 HoughLinesPointSet
    imgproc::hough_lines_point_set(
        &points,
        &mut lines,
        20,  // 最小投票数
        1,    // 投票累加器阈值
        rho_min,
        rho_max,
        rho_step,
        theta_min,
        theta_max,
        theta_step,
    )?;

    // 将结果复制到 line3d
    lines.copy_to(&mut line3d)?;
    // 输出结果
    for i in 0..line3d.len() {
        let line = line3d.get(i)?;
        println!(
            "votes: {}, rho: {}, theta: {}",
            line[0] as i32,
            line[1],
            line[2]
        );
    }

    Ok(())
}