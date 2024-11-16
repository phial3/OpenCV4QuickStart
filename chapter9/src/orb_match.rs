use anyhow::{Result, Error, Context};
use opencv::{
    core::{Mat, Point, Point2f, Scalar, RNG, DMatch,Ptr, Vector, KeyPoint},
    imgcodecs,
    imgproc,
    highgui,
    features2d::{self, Feature2D, DrawMatchesFlags, ORB, BFMatcher,ORB_ScoreType},
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

    // 特征点匹配
    let mut matches = Vector::<DMatch>::new();
    //定义特征点匹配的类，使用汉明距离
    let mut matcher = BFMatcher::create(opencv::core::NORM_HAMMING, false)?;
    //进行特征点匹配
    matcher.match_(&descriptions1, &mut matches, &opencv::core::no_array())?;
    matcher.match_(&descriptions2, &mut matches, &opencv::core::no_array())?;
    println!("matches={}", matches.len());

    // 通过汉明距离筛选匹配结果
    let mut min_dist = f64::MAX;
    let mut max_dist = 0f64;

    for dmatch in matches.iter() {
        let dist = dmatch.distance as f64;
        min_dist = min_dist.min(dist);
        max_dist = max_dist.max(dist);
    }

    println!("min_dist={}", min_dist);
    println!("max_dist={}", max_dist);

    // 将汉明距离较大的匹配点对删除
    let mut good_matches = Vector::new();
    for dmatch in matches.iter() {
        if (dmatch.distance as f64) <= f64::max(2.0 * min_dist, 20.0) {
            good_matches.push(dmatch.clone());
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
    highgui::imshow("最小汉明距离筛选", &outimg1)?;
    highgui::wait_key(0)?;

    Ok(())
}

fn orb_features(gray: &Mat, keypoints: &mut Vector<KeyPoint>, descriptions: &mut Mat) -> Result<()> {
    let mut orb = ORB::create(1000, 1.2f32, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 20)?;
    orb.detect(&gray, keypoints, &opencv::core::no_array())?;
    orb.compute(&gray, keypoints, descriptions)?;
    Ok(())
}
