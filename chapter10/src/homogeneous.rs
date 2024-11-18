use anyhow::{Result, Error, Context};
use image::open;
use opencv::{
    prelude::*,
    core::{Mat, Point2f, Point3f, Vec4f, Scalar, Size, Vector, TermCriteria},
    imgcodecs,
    imgproc,
    highgui,
    calib3d,
};

const BASE_PATH: &str = "../data/chapter3/";

pub(crate) fn run() -> Result<()> {
    // 设置两个三维坐标
    let mut points3 = Vector::from_slice(&[
        Point3f::new(3.0, 6.0, 1.5),
        Point3f::new(23.0, 32.0, 1.0),
    ]);

    // 非齐次坐标转齐次坐标
    let mut points4 = Mat::default();
    calib3d::convert_points_to_homogeneous(&points3, &mut points4).context("转换非齐次坐标到齐次坐标失败").unwrap();

    // 齐次坐标转非齐次坐标
    let mut points2 = Vector::<Point2f>::new();
    calib3d::convert_points_from_homogeneous(&points3, &mut points2).context("转换齐次坐标到非齐次坐标失败").unwrap();

    // 输出齐次坐标转非齐次坐标
    println!("***********齐次坐标转非齐次坐标*************");
    for (i, point) in points3.iter().enumerate() {
        println!("齐次坐标：{:?}", point);
        println!("非齐次坐标：{:?}", points2.get(i).unwrap());
    }

    // 输出非齐次坐标转齐次坐标
    println!("***********非齐次坐标转齐次坐标*************");
    for (i, point) in points3.iter().enumerate() {
        let homogeneous_point: &Vec4f = points4.at_2d(i as i32, 0)?;
        println!("齐次坐标：{:?}", point);
        println!("非齐次坐标：{:?}", homogeneous_point);
    }

    Ok(())
}

