use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Point2f, Scalar, RNG, DMatch, Ptr, Vector, KeyPoint},
    imgcodecs,
    imgproc,
    highgui,
    features2d::{self, Feature2D, FlannBasedMatcher, DrawMatchesFlags, ORB, BFMatcher, ORB_ScoreType},
    prelude::*,
};

const BASE_PATH: &str = "../data/chapter9/";

pub(crate) fn run() -> Result<()> {
    // 读取图像
    let img1 = imgcodecs::imread(&format!("{}box.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    let img2 = imgcodecs::imread(&format!("{}box_in_scene.png", BASE_PATH), imgcodecs::IMREAD_COLOR)?;
    if img1.empty() || img2.empty() {
        panic!("读取图像错误，请确认图像文件是否正确");
    }

    // 提取ORB特征点
    let mut keypoints1 = Vector::new();
    let mut keypoints2 = Vector::new();
    let mut descriptions1 = Mat::default();
    let mut descriptions2 = Mat::default();

    // 计算特征点
    orb_features(&img1, &mut keypoints1, &mut descriptions1)?;
    orb_features(&img2, &mut keypoints2, &mut descriptions2)?;

    // 判断描述子数据类型，如果数据类型不符需要进行类型转换
    let mut converted_desc1 = Mat::default();
    let mut converted_desc2 = Mat::default();

    if descriptions1.typ() != opencv::core::CV_32F && descriptions2.typ() != opencv::core::CV_32F {
        descriptions1.convert_to(&mut converted_desc1, opencv::core::CV_32F, 1.0, 0.0)?;
        descriptions2.convert_to(&mut converted_desc2, opencv::core::CV_32F, 1.0, 0.0)?;
    } else {
        converted_desc1 = descriptions1.clone();
        converted_desc2 = descriptions2.clone();
    }

    println!("keypoints1={}", keypoints1.len());
    println!("keypoints2={}", keypoints2.len());
    println!("descriptions1={:?}", descriptions1.size()?);
    println!("descriptions2={:?}", descriptions2.size()?);
    println!("converted_desc1={:?}", converted_desc1.size()?);
    println!("converted_desc2={:?}", converted_desc2.size()?);

    // 特征点匹配
    let mut matches = Vector::<DMatch>::new();
    let mut matcher = FlannBasedMatcher::create()?;
    matcher.match_(&converted_desc1, &mut matches, &opencv::core::no_array())?;
    matcher.match_(&converted_desc2, &mut matches, &opencv::core::no_array())?;
    println!("matches={}", matches.len());

    // 寻找距离最大值和最小值
    let mut min_dist = 100f64;
    let mut max_dist = 0f64;

    // FIXME: Error { code: "StsOutOfRange, -211", message: "Index: 0 out of bounds: 0..0" }
    for i in 0..converted_desc1.rows() {
        let dist = matches.get(i as usize).unwrap().distance as f64;
        min_dist = min_dist.min(dist);
        max_dist = max_dist.max(dist);
    }

    println!("Max dist: {}", max_dist);
    println!("Min dist: {}", min_dist);

    // 将最大值距离的0.4倍作为最优匹配结果进行筛选
    let mut good_matches = Vector::new();
    for i in 0..converted_desc1.rows() {
        if (matches.get(i as usize)?.distance as f64) < 0.4 * max_dist {
            good_matches.push(matches.get(i as usize)?);
        }
    }
    println!("good_matches={}", good_matches.len());

    // 绘制匹配结果
    let mut outimg = Mat::default();
    let mut outimg1 = Mat::default();

    features2d::draw_matches(
        &img1, &keypoints1,
        &img2, &keypoints2,
        &matches, &mut outimg,
        Scalar::all(-1.0),
        Scalar::all(-1.0),
        &Vector::new(),
        DrawMatchesFlags::DEFAULT,
    )?;

    features2d::draw_matches(
        &img1, &keypoints1,
        &img2, &keypoints2,
        &good_matches, &mut outimg1,
        Scalar::all(-1.0),
        Scalar::all(-1.0),
        &Vector::new(),
        DrawMatchesFlags::DEFAULT,
    )?;

    highgui::imshow("未筛选结果", &outimg)?;
    highgui::imshow("筛选结果", &outimg1)?;

    highgui::wait_key(0)?;

    Ok(())
}

fn orb_features(gray: &Mat, keypoints: &mut Vector<KeyPoint>, descriptions: &mut Mat) -> Result<()> {
    let mut orb = ORB::create(1000, 1.2f32, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 20)?;
    orb.detect(&gray, keypoints, &opencv::core::no_array())?;
    orb.compute(&gray, keypoints, descriptions)?;
    Ok(())
}