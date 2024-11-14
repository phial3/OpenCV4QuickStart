use anyhow::{Context, Result, Error};
use opencv::{
    core::{Mat, Point, Size},
    prelude::*,
};

pub(crate) fn run() -> Result<()> {
    // 定义数组并初始化
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0];

    // 创建单通道矩阵
    let img = unsafe { Mat::new_rows_cols_with_data_unsafe_def(3, 4, opencv::core::CV_32FC1, a.as_ptr() as *mut _)? };
    // 创建多通道矩阵
    let imgs = unsafe { Mat::new_rows_cols_with_data_unsafe_def(2, 3, opencv::core::CV_32FC2, a.as_ptr() as *mut _)? };

    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_idx = Point::new(0, 0);
    let mut max_idx = Point::new(0, 0);

    // 寻找单通道矩阵中的最值
    opencv::core::min_max_loc(&img, Some(&mut min_val), Some(&mut max_val), Some(&mut min_idx), Some(&mut max_idx), &Mat::default())?;
    println!("img中最大值是：{}  在矩阵中的位置: {:?}", max_val, max_idx);
    println!("img中最小值是：{}  在矩阵中的位置: {:?}", min_val, min_idx);

    // 将多通道矩阵展平为单通道矩阵
    let imgs_re = imgs.reshape(1, 4)?;

    // 寻找多通道矩阵中的最值
    opencv::core::min_max_loc(&imgs_re, Some(&mut min_val), Some(&mut max_val), Some(&mut min_idx), Some(&mut max_idx), &Mat::default())?;
    println!("imgs中最大值是：{}  在矩阵中的位置: {:?}", max_val, max_idx);
    println!("imgs中最小值是：{}  在矩阵中的位置: {:?}", min_val, min_idx);

    Ok(())
}