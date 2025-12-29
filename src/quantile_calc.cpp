#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath> // 用于round函数，优化分位数索引计算
using namespace Rcpp;

//' 【核心逻辑版】Rcpp分位数计算（无加速，仅保留0.01、0.5、0.99分位数计算）
//' @name cpp_quantile_calc  // 新增：补充@name字段，与函数名严格一致
//' @param df 数据框（仅包含数值列）
//' @param feature_cols 需计算分位数的列名（字符向量）
//' @return 列表：各列的0.01、0.5、0.99分位数（命名为q01, q50, q99，增强可读性）
//' @export
// [[Rcpp::export]] // 关键：保留默认的Rcpp导出宏，不添加冗余的name参数（自动生成导出文件的核心）
List cpp_quantile_calc(DataFrame df, CharacterVector feature_cols) {
  // 初始化结果存储，长度与特征列一致
  List result(feature_cols.size());
  // 为结果列表设置列名，与输入的特征列名对应
  result.names() = feature_cols;

  // 遍历每个特征列（保留核心的列遍历逻辑）
  for (int j = 0; j < feature_cols.size(); j++) {
    // 【关键修复1】将Rcpp的CharacterVector元素（proxy对象）转换为std::string，避免索引异常
    std::string col_name = as<std::string>(feature_cols[j]);
    // 用标准字符串作为列名提取数据列，确保能正确索引数据框的列
    NumericVector col = df[col_name]; 
    
    // 保留原本的缺失值过滤逻辑
    col = col[!is_na(col)]; 
    
    // 保留原本的空值处理逻辑：如果列无有效数据，返回NA
    if (col.size() == 0) {
      // 为NA结果添加命名，与正常结果格式统一
      NumericVector na_vec = NumericVector::create(NA_REAL, NA_REAL, NA_REAL);
      na_vec.names() = CharacterVector::create("q01", "q50", "q99");
      result[j] = na_vec;
      continue;
    }

    // 移除批次加速逻辑，仅保留核心的排序逻辑
    std::sort(col.begin(), col.end());
    int n = col.size(); // 有效数据的长度

    // 【关键修复2】优化分位数索引计算：使用round()处理小数部分，同时限制索引在0~n-1范围内（避免越界）
    int idx01 = std::max(0, std::min(static_cast<int>(std::round(n * 0.01)), n - 1));
    int idx50 = std::max(0, std::min(static_cast<int>(std::round(n * 0.5)), n - 1));
    int idx99 = std::max(0, std::min(static_cast<int>(std::round(n * 0.99)), n - 1));

    // 提取分位数数值
    double q01 = col[idx01];
    double q50 = col[idx50];
    double q99 = col[idx99];

    // 为结果添加命名，增强可读性（非必须，但提升用户体验）
    NumericVector q_vec = NumericVector::create(q01, q50, q99);
    q_vec.names() = CharacterVector::create("q01", "q50", "q99");
    result[j] = q_vec;
  }

  return result;
}