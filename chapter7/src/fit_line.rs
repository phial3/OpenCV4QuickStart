use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Scalar, Size, Vec4i, Vec4f, Point2f, Vector},
    imgcodecs,
    imgproc,
    highgui,
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter7/";

pub(crate) fn run() -> Result<()> {
    let mut lines = Vec4f::default(); // 存放拟合后的直线
    let mut points = Vector::<Point2f>::new(); // 待检测是否存在直线的所有点

    const POINTS: [[f32; 2]; 20] = [
        [0.0, 0.0],
        [10.0, 11.0],
        [21.0, 20.0],
        [30.0, 30.0],
        [40.0, 42.0],
        [50.0, 50.0],
        [60.0, 60.0],
        [70.0, 70.0],
        [80.0, 80.0],
        [90.0, 92.0],
        [100.0, 100.0],
        [110.0, 110.0],
        [120.0, 120.0],
        [136.0, 130.0],
        [138.0, 140.0],
        [150.0, 150.0],
        [160.0, 163.0],
        [175.0, 170.0],
        [181.0, 180.0],
        [200.0, 190.0],
    ];
    // 将所有点存放在 Vector 中
    for i in 0..20 {
        points.push(Point2f::new(POINTS[i][0], POINTS[i][1]));
    }

    // 参数设置
    let param = 0.0; // 距离模型中的数值参数 C
    let reps = 0.01; // 坐标原点与直线之间的距离精度
    let aeps = 0.01; // 角度精度

    // 拟合直线
    imgproc::fit_line(&points, &mut lines, imgproc::DIST_L1, param, reps, aeps)?;
    let k = lines[1] / lines[0]; // 直线斜率
    println!("直线斜率：k={}", k);
    println!("直线上一点坐标 x：{}, y：{}", lines[2], lines[3]);
    println!("直线解析式：y={}*(x-{:.2})+{:.2}", k, lines[2], lines[3]);

    Ok(())
}
