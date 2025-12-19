#' @useDynLib SA25204160
#' @importFrom Rcpp sourceCpp evalCpp
#' @importFrom dplyr select all_of %>%
NULL

#' Calculate Quantiles
#'
#' This function calculates quantiles using Rcpp.
#'
#' @param df DataFrame
#' @param feature_cols Character vector of column names
#' @return List of quantiles
#' @export
cpp_quantile_calc <- function(df, feature_cols) {
  # This is just a placeholder - the actual implementation is in C++
}

#' MLP Residual Model Training for Financial Data
#'
#' This function trains a residual MLP model for financial data.
#' Note: The torch package is required for this function to work.
#'
#' @param preprocessed_data Preprocessed dataset (data frame, contains feature columns and target column)
#' @param target_col Target column name (character)
#' @return Trained model object
#' @export
train_residual_mlp <- function(preprocessed_data, target_col) {
  # Check if torch package is available using requireNamespace
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("The 'torch' package is required but not installed. ",
         "Please install it with: install.packages('torch')")
  }

  # Check if dplyr package is available
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("The 'dplyr' package is required but not installed. ",
         "Please install it with: install.packages('dplyr')")
  }

  # Use torch:: prefix for all torch functions
  features <- preprocessed_data %>%
    dplyr::select(-dplyr::all_of(target_col)) %>%
    as.matrix() %>%
    torch::torch_tensor(dtype = torch::torch_float32())

  target <- preprocessed_data[[target_col]] %>%
    as.matrix() %>%
    torch::torch_tensor(dtype = torch::torch_float32())

  # Define residual block: keep the original residual connection core logic
  ResidualBlock <- torch::nn_module(
    initialize = function(in_features, out_features) {
      self$linear1 <- torch::nn_linear(in_features, out_features)
      self$linear2 <- torch::nn_linear(out_features, out_features)
      self$relu <- torch::nn_relu()
      # Keep the original residual adaptation logic
      if (in_features != out_features) {
        self$shortcut <- torch::nn_linear(in_features, out_features)
      } else {
        self$shortcut <- torch::nn_identity()
      }
    },
    forward = function(x) {
      # Keep the original residual calculation core logic
      residual <- self$shortcut(x)
      x <- self$linear1(x) %>% self$relu()
      x <- self$linear2(x)
      x + residual %>% self$relu()
    }
  )

  # Define residual MLP model: keep the original model structure logic
  ResidualMLP <- torch::nn_module(
    initialize = function(n_features) {
      self$block1 <- ResidualBlock(n_features, 64)
      self$block2 <- ResidualBlock(64, 32)
      self$output <- torch::nn_linear(32, 1)
    },
    forward = function(x) {
      # Keep the original forward propagation logic
      x %>% self$block1() %>% self$block2() %>% self$output()
    }
  )

  # Initialize model and optimizer: keep the original initialization logic
  model <- ResidualMLP(ncol(features))
  optimizer <- torch::optim_adam(model$parameters) # Use default parameters, no acceleration
  loss_fn <- torch::nn_mse_loss()

  # Training loop: keep the original training core logic (fixed 10 epochs)
  for (epoch in 1:10) {
    # Keep the original forward propagation logic
    y_pred <- model(features)
    loss <- loss_fn(y_pred, target)

    # Keep the original backward propagation logic
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
  }

  # Keep the original model return logic
  return(model)
}
