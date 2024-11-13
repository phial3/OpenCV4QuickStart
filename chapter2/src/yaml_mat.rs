use serde::{Deserialize, Serialize};
use std::fs;
use anyhow::{Result, Error, Context};

#[derive(Debug, Serialize, Deserialize)]
pub struct YamlConfig {
    count: u32,
    name: String,
    version: String,
    dependencies: Vec<String>,
}

/// 因为 Mat 不能被序列化，所以需要自定义一个结构体来保存 Mat 对象的关键信息
/// 并实现 Serialize 和 Deserialize 特征
/// 定义一个结构体，表示保存 Mat 对象的关键信息
#[derive(Serialize, Deserialize, Debug)]
struct MatData {
    rows: i32,
    cols: i32,
    type_: i32,
    data: Vec<u8>,
}

fn read_yaml(file_path: &str) -> Result<YamlConfig, Error> {
    // 读取文件内容
    let content = fs::read_to_string(file_path)?;

    // 反序列化 YAML 内容为 Rust 结构体
    let config: YamlConfig = serde_yml::from_str(&content)?;

    Ok(config)
}

fn write_yaml(file_path: &str, config: &YamlConfig) -> Result<(), Error> {
    // 将 Rust 结构体序列化为 YAML 字符串
    let yaml = serde_yml::to_string(config)?;

    // 写入 YAML 到文件
    fs::write(file_path, yaml)?;

    Ok(())
}

pub(crate) fn run() -> Result<()> {
    // 1. 写入 YAML 到文件
    let config = YamlConfig {
        count: 4,
        name: "Rust".to_string(),
        version: "1.0".to_string(),
        dependencies: vec!["serde".to_string(), "serde_yaml".to_string()],
        // mat: Mat::new_rows_cols(3, 3, opencv::core::CV_8U)?,
    };

    // 写入 YAML 文件
    write_yaml("config.yaml", &config)?;

    // 2. 读取 YAML 文件内容
    let config = read_yaml("./config.yaml")?;
    println!("{:?}", config);

    Ok(())
}