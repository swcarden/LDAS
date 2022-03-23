# Load libraries ----
library(tidyverse)
library(sfsmisc)
source("LDAS functions.R")

# Creates the figure visualizing a Halton and pseudo-random sequence ----
pdf("images/halton-v-PR.pdf", width = 10, height = 5)
n <- 300
halton <- QUnif(n, p = 2)
data.frame("X" = c(halton[ , 1], runif(n)),
           "Y" = c(halton[ , 2], runif(n)),
           "Method" = c(rep("Halton", n), rep("Pseudo-random", n))) %>% 
  ggplot(aes(X, Y)) +
  geom_point() + 
  facet_grid(cols = vars(Method)) +
  theme_minimal()
dev.off()

# Creates data/figure w/ disc comparisons of Halton and pseudo-random sequences ----
# Create data
num_runs <- 10
points_per_run <- 500

halton_list <- vector("list", num_runs)
unif_list <- vector("list", num_runs)
for (index in 1:num_runs) {
  print(index / num_runs * 100)
  halton_list[[index]] <- QUnif(points_per_run, p = 2) %>%
    data.frame(
      "Run" = rep(index, points_per_run),
      "Method" = rep("Halton", points_per_run)
    )
  unif_list[[index]] <- runif(points_per_run * 2) %>%
    matrix(ncol = 2) %>%
    data.frame(
      "Run" = rep(index, points_per_run),
      "Method" = rep("Unif", points_per_run)
    )
  index <- index + 1
}

list_all <- c(halton_list, unif_list)

df_all <- bind_rows(list_all) %>% 
  cbind(data.frame("Disc_CV" = 0,
                   "Disc_Classic" = 0,
                   "Step" = 0))

# Go through each run, and estimate the max discrepancy
# at each point in time.  Save these estimates in a new column.
for (i in 1:nrow(df_all)) {
  print(i / nrow(df_all) * 100)
  temp_df <- df_all %>% 
    filter(Run == df_all[i, "Run"], 
           Method == df_all[i, "Method"])
  temp_df_index <- (i - 1) %% points_per_run + 1
  if (temp_df_index > 1) {
    df_all[i, "Disc_CV"] <- 
      estimate_discrepancy_CV(temp_df[1:temp_df_index, 
                                      1:2, 
                                      drop = FALSE])
    df_all[i, "Disc_Classic"] <- 
      estimate_discrepancy_classic(temp_df[1:temp_df_index, 
                                           1:2, 
                                           drop = FALSE])$max
    df_all[i, "Step"] <- temp_df_index
  }
}

# Save file so above stuff doesn't have to run again
write.csv(df_all, "data/halton_v_unif.csv")

# Create figure 
df_all <- read_csv("data/halton_v_unif.csv") %>% 
  rename("Sequence Type" = Method) %>% 
  mutate(`Sequence Type` = case_when(
    `Sequence Type` == "Unif" ~ "Pseudo-random",
    TRUE ~ `Sequence Type`))

pdf("images/halton-v-unif-disc.pdf", width = 10, height = 5)
df_all %>%
  filter(Step >= 5) %>% 
  group_by(Step, `Sequence Type`) %>% 
  summarise(Classic = mean(Disc_Classic),
            CV = mean(Disc_CV),
            .groups = "keep") %>%
  pivot_longer(c("Classic", "CV"), 
               values_to = "Discrepancy", 
               names_to = "Discrepancy Method") %>% 
  ggplot(aes(x = Step, y = Discrepancy, color = `Sequence Type`)) +
  geom_smooth() +
  facet_wrap(facets = vars(`Discrepancy Method`), scales = "free_y") +
  labs(x = "N", y = "Average Discrepancy") +
  theme_minimal()
dev.off()

# Create data/figure for LDAS and psuedo-random visualization ----
# Produce a comparison of discrepancy measures for paper ----
ldas_config <- list(
  "state_dim" = 1,
  "action_dim" = 1,
  "learning_rate" = .001,
  "learning_rate_decay" = .95,
  "num_recent_actions" = 5,
  "num_candidate_actions" = 10,
  "rel_threshold" = .001
)

sa_hist_LDAS <- simulate_ldas_phantom(n = 300, ldas_config)
sa_hist_LDAS <- sa_hist_LDAS %>% 
  as_tibble %>% 
  rename(X = V1, Y = V2) %>% 
  mutate(Method = "LDAS")

sa_hist_unif <- runif(300 * 2) %>% 
  matrix(ncol = 2) %>% 
  as_tibble %>% 
  rename(X = V1, Y = V2) %>% 
  mutate(Method = "Pseudo-random")

sa_hist <- bind_rows(sa_hist_LDAS, sa_hist_unif) 

pdf("images/ldas-v-unif.pdf", width = 10, height = 5)
ggplot(sa_hist, aes(X, Y)) +
  facet_grid(cols = vars(Method)) +
  geom_point(aes(X, Y)) + 
  xlab("") +
  ylab("") +
  theme_minimal() 
dev.off()

# Multi-dim LDAS data/figure ----
# Create data.  Takes a long time.
max_state_dims <- 3
max_action_dims <- 3
num_runs_per_dim <- 100
points_per_run <- 100
index <- 1
num_runs_total <- num_runs_per_dim * max_state_dims * max_action_dims
sa_hist_LDAS_list <- vector("list", num_runs_total)
sa_hist_unif_list <- vector("list", num_runs_total)
for (d_s in 1:max_state_dims){
  for (d_a in 1:max_action_dims){
    ldas_config <- list(
      "state_dim" = d_s,
      "action_dim" = d_a,
      "learning_rate" = .001,
      "learning_rate_decay" = .95,
      "num_recent_actions" = 5,
      "num_candidate_actions" = 10,
      "rel_threshold" = .001
    )
    for (i in 1:num_runs_per_dim) {
      print(index / num_runs_total * 100)
      sa_hist_LDAS_list[[index]] <- simulate_ldas_phantom(n = points_per_run, 
                                                          ldas_config) %>% 
        data.frame("Run" = rep(i, points_per_run),
                   "Method" = rep("LDAS", points_per_run),
                   "State_Dims" = rep(d_s, points_per_run),
                   "Action_Dims" = rep(d_a, points_per_run))
      sa_hist_unif_list[[index]] <- runif(points_per_run*2) %>% matrix(ncol = 2) %>% 
        data.frame("Run" = rep(i, points_per_run),
                   "Method" = rep("Unif", points_per_run),
                   "State_Dims" = rep(d_s, points_per_run),
                   "Action_Dims" = rep(d_a, points_per_run))
      index <- index + 1
    }
  }
}

sa_hist_list_all <- c(sa_hist_LDAS_list, sa_hist_unif_list)
#sa_hist_all <- do.call("rbind", sa_hist_list_all)

sa_hist_all <- bind_rows(sa_hist_list_all)

sa_hist_all <- cbind(sa_hist_all, 
                     data.frame("Disc" = rep(0, nrow(sa_hist_all)),
                                "Time" = rep(0, nrow(sa_hist_all)))
)

# Reorder columns
sa_hist_all <- sa_hist_all %>% 
  relocate(Run, .after = last_col()) %>%
  relocate(State_Dims, .after = last_col()) %>% 
  relocate(Action_Dims, .after = last_col()) %>%
  relocate(Method, .after = last_col())

# Go through each run, and estimate the max discrepancy
# at each point in time.  Save these estimates in a new column.
for (i in 1:nrow(sa_hist_all)) {
  print(i / nrow(sa_hist_all) * 100)
  temp_df <- sa_hist_all %>% 
    filter(Run == sa_hist_all[i, "Run"], 
           Method == sa_hist_all[i, "Method"],
           State_Dims == sa_hist_all[i, "State_Dims"],
           Action_Dims == sa_hist_all[i, "Action_Dims"])
  temp_df_index <- (i - 1) %% points_per_run + 1
  if (temp_df_index > 1) {
    sa_hist_all[i, "Disc"] <- 
      estimate_discrepancy_CV(temp_df[1:temp_df_index, 
                                      1:(sa_hist_all[i, "State_Dims"] + sa_hist_all[i, "Action_Dims"]), 
                                      drop = FALSE])
    sa_hist_all[i, "Time"] <- temp_df_index
  }
}
write.csv(sa_hist_all, "ldas_multidim_experiment.csv")

# Create figure 
sa_hist_all <- read.csv("data/multi_dim_disc.csv")
pdf("images/multi-dim.pdf")
sa_hist_all %>%
  filter(Time >= 5) %>% 
  mutate(Action_Dims = recode(Action_Dims, `1` = "1-Action Dimension", `2` = "2-Action Dimensions", `3` = "3-Action Dimensions")) %>% 
  mutate(State_Dims = recode(State_Dims, `1` = "1-State Dimension", `2` = "2-State Dimensions", `3` = "3-State Dimensions")) %>% 
  mutate(State_Dims = as.factor(State_Dims),
         Action_Dims = as.factor(Action_Dims)) %>% 
  group_by(Time, Method, State_Dims, Action_Dims) %>% 
  summarise(Mean_Disc = mean(Disc), .groups = "keep") %>%
  #ungroup() %>% 
  ggplot(aes(x = Time, y = Mean_Disc, color = Method)) +
  labs(y= "Mean Discrepancy Estimate", x = "Number of Steps") +
  facet_grid(rows = vars(State_Dims), cols = vars(Action_Dims)) +
  geom_smooth() +
  theme_minimal()
dev.off()

# Gym environment figures ----
# These require running the three python scripts 
# and "gym_env_disc_analysis.R" first
# Import data 
mcar <- read.csv("data/mcar_disc.csv") %>% 
  mutate("Experiment" = "Mountain Car")
lunar <- read.csv("data/lunar_disc.csv") %>% 
  mutate("Experiment" = "Lunar Lander")
rcar <- read.csv("data/rcar_disc.csv") %>% 
  mutate("Experiment" = "Race Car")

gym_disc_df <- rbind(
  mcar %>%
    select(-Position,-Velocity,-Action),
  lunar %>%
    select(
      -Pos_x,
      -Pos_y,
      -Vel_x,
      -Vel_y,
      -Angle,
      -Angle_Vel,-Left_Leg,
      -Right_Leg,
      -Main_Engine,
      -Side_Thrusters
    ),
  rcar
)

# Gym environment discrepancy
pdf("images/disc-results.pdf", width = 12, height = 5)
gym_disc_df %>%
  mutate(Method = case_when(
    Method == "Unif" ~ "PR",
    TRUE ~ Method)) %>% 
  filter(Step >= 5) %>%
  mutate(Experiment = factor(Experiment, levels = c("Mountain Car", "Lunar Lander", "Race Car"))) %>% 
  group_by(Step, Method, Experiment) %>%
  summarise(Mean_Disc = mean(Disc), .groups = "keep") %>%
  ggplot(aes(x = Step, y = Mean_Disc, color = Method, linetype = Method)) +
  labs(y = "Mean Discrepancy Estimate", x = "Number of Steps") +
  facet_wrap(facets = vars(Experiment), scales = "free_y") +
  geom_smooth() +
  theme_minimal()
dev.off()

# Gym environment time
pdf("images/time-results.pdf", width = 12, height = 5)
gym_disc_df %>%
  filter(Step >= 5) %>%
  mutate(Method = case_when(
    Method == "Unif" ~ "PR",
    TRUE ~ Method)) %>% 
  mutate(Experiment = factor(Experiment, levels = c("Mountain Car", "Lunar Lander", "Race Car"))) %>% 
  group_by(Step, Method, Experiment) %>%
  summarise(Mean_time = mean(Time_Elapsed), .groups = "keep") %>%
  ggplot(aes(x = Step, y = Mean_time, color = Method)) +
  labs(y = "Mean Time Elapsed in Seconds", x = "Number of Steps") +
  facet_wrap(facets = vars(Experiment), scales = "free_y") +
  geom_smooth() +
  theme_minimal()
dev.off()

