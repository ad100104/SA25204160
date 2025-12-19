#' 【核心逻辑版】金融数据MLP残差模型训练（无加速，仅保留核心训练逻辑）
#' @param preprocessed_data 预处理后的数据集（数据框，包含特征列和目标列）
#' @param target_col 目标列名（字符）
#' @return 训练好的模型对象（保留原本的模型返回逻辑）
#' @export
train_residual_mlp <- function(preprocessed_data, target_col) {
  # 改用require检查并加载包，更规范（替代library）
  if (!require(torch, quietly = TRUE)) {
    stop("需要安装torch包：install.packages('torch')")
  }

  # 关键修复：用dplyr::select显式调用，避免找不到函数
  features <- preprocessed_data %>%
    dplyr::select(-dplyr::all_of(target_col)) %>%
    as.matrix() %>%
    torch_tensor(dtype = torch_float32())

  target <- preprocessed_data[[target_col]] %>%
    as.matrix() %>%
    torch_tensor(dtype = torch_float32())

  # 定义残差块：保留原本的残差连接核心逻辑
  ResidualBlock <- nn_module(
    initialize = function(in_features, out_features) {
      self$linear1 <- nn_linear(in_features, out_features)
      self$linear2 <- nn_linear(out_features, out_features)
      self$relu <- nn_relu()
      # 保留原本的残差适配逻辑
      if (in_features != out_features) {
        self$shortcut <- nn_linear(in_features, out_features)
      } else {
        self$shortcut <- nn_identity()
      }
    },
    forward = function(x) {
      # 保留原本的残差计算核心逻辑
      residual <- self$shortcut(x)
      x <- self$linear1(x) %>% self$relu()
      x <- self$linear2(x)
      x + residual %>% self$relu()
    }
  )

  # 定义残差MLP模型：保留原本的模型结构逻辑
  ResidualMLP <- nn_module(
    initialize = function(n_features) {
      self$block1 <- ResidualBlock(n_features, 64)
      self$block2 <- ResidualBlock(64, 32)
      self$output <- nn_linear(32, 1)
    },
    forward = function(x) {
      # 保留原本的前向传播逻辑
      x %>% self$block1() %>% self$block2() %>% self$output()
    }
  )

  # 初始化模型和优化器：保留原本的初始化逻辑
  model <- ResidualMLP(ncol(features))
  optimizer <- optim_adam(model$parameters) # 用默认参数，无加速
  loss_fn <- nn_mse_loss()

  # 训练循环：保留原本的训练核心逻辑（固定10轮）
  for (epoch in 1:10) {
    # 保留原本的前向传播逻辑
    y_pred <- model(features)
    loss <- loss_fn(y_pred, target)

    # 保留原本的反向传播逻辑
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
  }

  # 保留原本的模型返回逻辑
  return(model)
}
