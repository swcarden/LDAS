pause <- function() {
  if (interactive())
  {
    invisible(readline(prompt = "Press <Enter> to continue..."))
  }
  else
  {
    cat("Press <Enter> to continue...")
    invisible(readLines(file("stdin"), 1))
  }
}

get_closest_point <- function(X, x0) {
  distance <- X %>%
    sweep(2, x0) %>%
    abs %>%
    rowSums
  closest_index <- which.min(distance)
  closest_distance <- distance[closest_index]
  closest_point <- X[closest_index,]
  return_obj <- list("distance" = closest_distance,
                     "point" = closest_point)
}

get_norm <- function(x) {
  sqrt(sum(x ^ 2))
}

get_phantom_matrix <- function(s, a, eps) {
  d <- length(a)
  phantom <- matrix(a,
                    nrow = 2 * d,
                    ncol = d,
                    byrow = TRUE)
  indices <- seq(1, 2 * d ^ 2 - 1, 2 * (d + 1))
  phantom[indices] <- -eps
  phantom[indices + 1] <- 1 + eps
  state_portion <-
    matrix(s,
           nrow = 2 * d,
           ncol = length(s),
           byrow = TRUE)
  cbind(state_portion, phantom)
}

get_ld_action <- function(X, s, ldas_config) {
  with(ldas_config,
       {
         sa_dim <- state_dim + action_dim
         #best_action <- rep(NA, action_dim)
         best_distance <- 0
         for (i in 1:num_candidate_actions) {
           keep_going <- TRUE
           temp_lr <- learning_rate
           a <- runif(action_dim)
           recent_actions <- vector("list", num_recent_actions)
           recent_actions[[1]] <- a
           for (j in 2:num_recent_actions) {
             # I think phantom points need to be added here
             phantom <-
               get_phantom_matrix(s, a, 1 / nrow(X) ^ (1 / action_dim))
             temp_X <- X %>%
               rbind(phantom)
             closest_point <- get_closest_point(temp_X, c(s, a))$point
             diff <- a - closest_point[(state_dim + 1):sa_dim]
             a <- a + temp_lr * diff / get_norm(diff)
             temp_lr <- temp_lr * learning_rate_decay
             a[a < 0] <- 0
             a[a > 1] <- 1
             if (all(a == 0 | a == 1)) {
               keep_going <- FALSE
               break
             }
             else if (temp_lr < rel_threshold) {
               keep_going <- FALSE
               break
             }
             else{
               recent_actions[[j]] <- a
             }
           }
           while (keep_going) {
             phantom <- get_phantom_matrix(s, a, 1 / i ^ (1 / action_dim))
             temp_X <- X %>%
               rbind(phantom)
             closest_point <- get_closest_point(X, c(s, a))$point
             diff <- a - closest_point[(state_dim + 1):sa_dim]
             a <- a + temp_lr * diff / get_norm(diff)
             temp_lr <- temp_lr * learning_rate_decay
             a[a < 0] <- 0
             a[a > 1] <- 1
             if (all(a == 0 | a == 1)) {
               keep_going = FALSE
             }
             else if (sum(abs(a - recent_actions[[1]])) < threshold |
                      temp_lr < rel_threshold) {
               keep_going <- FALSE
             } else{
               recent_actions <- c(list(recent_actions[2:num_recent_actions]), 
                                   list(a))
             }
             #print(learning_rate)
           }
           closest_info <- get_closest_point(X, c(s, a))
           if (closest_info$distance > best_distance) {
             best_distance <- closest_info$distance
             best_action <- a
           }
         }
         return(best_action)
       })
}

estimate_discrepancy_CV <- function(X) {
  distances <- dist(X, method = "manhattan") %>% 
    as.matrix
  diag(distances) <- NA
  min_distances <- apply(distances, 1, min, na.rm = TRUE)
  sd(min_distances)/mean(min_distances)^2
}

estimate_discrepancy_classic <- function(X,
                                 num_rectangles = 500,
                                 plotstuff = FALSE) {
  # X is an n x d matrix.  Each row represents a point in d-dimensional space.
  #
  d <- ncol(X)
  n <- nrow(X)
  # Initialize vector counting number of point in rectangles
  num_points <- rep(0, num_rectangles)
  # Initiate vector with discrepancy per rectangle
  discrepancy <- rep(0, num_rectangles)
  
  max_discrepancy <- 0
  max_disc_rect <- NA
  
  for (j in 1:num_rectangles) {
    lower <- runif(d, min = 0, max = 1)
    dim_length <- runif(d, min = 0, max = 1)
    upper <- lower + dim_length
    # This is a vector of indices for points inside the rectangle
    is_inside <- 1:n
    temp_X <- X
    for (i in 1:d) {
      temp_X <- temp_X[is_inside, , drop = FALSE]
      x <- temp_X[, i]
      is_inside_dim_i <-
        which(((upper[i] < 1) & ((x > lower[i]) & (x < upper[i]))) |
                ((upper[i] > 1) &
                   ((x < (
                     upper[i] - 1
                   )) | (x > lower[i]))))
      is_inside <- is_inside[is_inside_dim_i]
    }
    num_points[j] <- length(is_inside)
    discrepancy[j] <- abs((num_points[j] / n) - prod(dim_length))
    
    if (discrepancy[j] > max_discrepancy) {
      max_discrepancy <- discrepancy[j]
      max_disc_rect <- rbind(lower, upper)
    }
    
  }
  if (plotstuff) {
    hist(discrepancy)
    discrepancy %>% summary %>% print
  }
  max(discrepancy)
  median_discrepancy <- median(discrepancy)
  return_obj <-
    list("max" = max_discrepancy,
         "median" = median_discrepancy,
         "rect" = max_disc_rect)
}

simulate_ldas_phantom <-
  function(n = 200,
           ldas_config,
           plotstuff = FALSE) {
    with(ldas_config, {
      sa_dim <- state_dim + action_dim
      threshold <- rel_threshold * action_dim
      if (plotstuff) {
        plot(
          NA,
          NA,
          xlim = c(0, 1),
          ylim = c(0, 1),
          xlab = "State space",
          ylab = "Action space"
        )
      }
      sa_hist <- matrix(0, nrow = n, ncol = sa_dim)
      for (i in 1:n) {
        s <- runif(state_dim)
        temp_lr <- learning_rate
        if (plotstuff) {
          plot(
            s,
            a,
            xlim = c(0, 1),
            ylim = c(0, 1),
            col = 'red',
            pch = 15,
            xlab = "State space",
            ylab = "Action space"
          )
        }
        if (i == 1) {
          a <- runif(action_dim)
          sa_hist[1,] <- c(s, a)
        }
        if (i > 1) {
          X <- matrix(sa_hist[1:(i - 1),], nrow = i - 1)
          a <- get_ld_action(X, s, ldas_config)
          sa_hist[i,] <- c(s, a)
          if (plotstuff) {
            plot(
              s,
              a,
              xlim = c(0, 1),
              ylim = c(0, 1),
              col = 'red',
              pch = 15,
              xlab = "State space",
              ylab = "Action space"
            )
            points(X)
            pause()
          }
        }
      }
      return(sa_hist)
    })
  }