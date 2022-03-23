source("LDAS functions.R")
library(dplyr)
library(magrittr)
library(ggplot2)

# MCar ----
mcar_unif_disc <- read.csv("data/mcar_uniform_disc.csv", header = FALSE)
names(mcar_unif_disc) <- c("Position", "Velocity", "Action", "Run", "Step", "Time_Elapsed")
mcar_unif_disc <- mcar_unif_disc %>% 
  mutate(Method = "Unif", Disc = NA) %>% 
  arrange(Run)

mcar_LDAS_disc <- read.csv("data/mcar_LDAS_disc.csv", header = FALSE)
names(mcar_LDAS_disc) <- c("Position", "Velocity", "Action", "Run", "Step", "Time_Elapsed")
mcar_LDAS_disc <- mcar_LDAS_disc %>% 
  mutate(Method = "LDAS", Disc = NA) %>% 
  arrange(Run)

mcar_OU_disc <- read.csv("data/mcar_OU_disc.csv", header = FALSE)
names(mcar_OU_disc) <- c("Position", "Velocity", "Action", "Run", "Step", "Time_Elapsed")
mcar_OU_disc <- mcar_OU_disc %>% 
  mutate(Method = "OU", Disc = NA) %>% 
  arrange(Run)

mcar_disc <- bind_rows(mcar_unif_disc, mcar_LDAS_disc, mcar_OU_disc)

# Discrepancy calculation
points_per_run <- mcar_disc %>% 
  filter(Run == 1, Method == "Unif") %>% 
  nrow
for (i in 1:nrow(mcar_disc)) {
  print(i / nrow(mcar_disc) * 100)
  temp_df <- mcar_disc %>% 
    filter(Run == mcar_disc[["Run"]][i], 
           Method == mcar_disc[["Method"]][i])
  temp_df_index <- (i - 1) %% points_per_run + 1
  if (temp_df_index > 1) {
    mcar_disc[i, "Disc"] <- 
      estimate_discrepancy_CV(temp_df[1:temp_df_index, 
                                      1:3, 
                                      drop = FALSE])
    # Error catch
    if( is.nan(as.numeric(mcar_disc[i, "Disc"]))) {
      print(i)
    }
  }
}

# Lunar ----
lunar_unif_disc <- read.csv("data/lunar_uniform_disc.csv", header = FALSE)
names(lunar_unif_disc) <- c("Pos_x", "Pos_y", "Vel_x", "Vel_y", 
                            "Angle", "Angle_Vel", "Left_Leg", "Right_Leg",
                            "Main_Engine", "Side_Thrusters","Run",
                            "Step", "Time_Elapsed")
lunar_unif_disc <- lunar_unif_disc %>% 
  mutate(Method = "Unif", Disc = NA) %>% 
  arrange(Run)

lunar_LDAS_disc <- read.csv("data/lunar_LDAS_disc.csv", header = FALSE)
names(lunar_LDAS_disc) <- c("Pos_x", "Pos_y", "Vel_x", "Vel_y", 
                            "Angle", "Angle_Vel", "Left_Leg", "Right_Leg",
                            "Main_Engine", "Side_Thrusters","Run",
                            "Step", "Time_Elapsed")
lunar_LDAS_disc <- lunar_LDAS_disc %>% 
  mutate(Method = "LDAS", Disc = NA) %>% 
  arrange(Run)

lunar_OU_disc <- read.csv("data/lunar_OU_disc.csv", header = FALSE)
names(lunar_OU_disc) <- c("Pos_x", "Pos_y", "Vel_x", "Vel_y", 
                            "Angle", "Angle_Vel", "Left_Leg", "Right_Leg",
                            "Main_Engine", "Side_Thrusters","Run",
                            "Step", "Time_Elapsed")
lunar_OU_disc <- lunar_OU_disc %>% 
  mutate(Method = "OU", Disc = NA) %>% 
  arrange(Run)

lunar_disc <- bind_rows(lunar_unif_disc, lunar_LDAS_disc, lunar_OU_disc)

# Discrepancy calculation
points_per_run <- lunar_disc %>% 
  filter(Run == 1, Method == "Unif") %>% 
  nrow
for (i in 1:nrow(lunar_disc)) {
  print(i / nrow(lunar_disc) * 100)
  temp_df <- lunar_disc %>% 
    filter(Run == lunar_disc[["Run"]][i], 
           Method == lunar_disc[["Method"]][i])
  temp_df_index <- (i - 1) %% points_per_run + 1
  if (temp_df_index > 1) {
    lunar_disc[i, "Disc"] <- 
      estimate_discrepancy_CV(temp_df[1:temp_df_index, 
                                      1:10, 
                                      drop = FALSE])
    # Error catch
    if( is.nan(as.numeric(lunar_disc[i, "Disc"]))) {
      print(i)
    }
  }
}

# RCar ----
# Data is much bigger, so run separately, 
# run garbage collector, then combine
# RCar unif ----
rcar_unif_disc <- read_csv("rcar_uniform_disc.csv", col_names = FALSE)
names(rcar_unif_disc) <- c(paste0("Pixel", 1:(96*96*3)),
                                 "Steer", "Gas", "Brake",
                                 "Run", "Step", "Time_Elapsed")
rcar_unif_disc <- rcar_unif_disc %>%
  mutate(Method = "Unif", Disc = NA) %>%
  arrange(Run)

# Discrepancy calculation
points_per_run <- rcar_unif_disc %>%
  filter(Run == 1) %>%
  nrow
for (i in 1:nrow(rcar_unif_disc)) {
  print(i / nrow(rcar_unif_disc) * 100)
  temp_df <- rcar_unif_disc %>%
    filter(Run == rcar_unif_disc[["Run"]][i])
  temp_df_index <- (i - 1) %% points_per_run + 1
  if (temp_df_index > 1) {
    rcar_unif_disc[i, "Disc"] <-
      estimate_discrepancy_CV(temp_df[1:temp_df_index,
                                      1:(96*96*3+3),
                                      drop = FALSE])
    # Error catch
    if( is.nan(as.numeric(rcar_unif_disc[i, "Disc"]))) {
      print(i)
    }
  }
}

rcar_unif_disc <- rcar_unif_disc[ , (96*96*3 + 3 + 1):ncol(rcar_unif_disc)]
gc()

# RCar LDAS ----
rcar_ldas_disc <- read_csv("rcar_ldas_disc.csv", col_names = FALSE)
names(rcar_ldas_disc) <- c(paste0("Pixel", 1:(96*96*3)),
                           "Steer", "Gas", "Brake",
                           "Run", "Step", "Time_Elapsed")
rcar_ldas_disc <- rcar_ldas_disc %>%
  mutate(Method = "ldas", Disc = NA) %>%
  arrange(Run)

# Discrepancy calculation
points_per_run <- rcar_ldas_disc %>%
  filter(Run == 1) %>%
  nrow

for (i in 1:nrow(rcar_ldas_disc)) {
  print(i / nrow(rcar_ldas_disc) * 100)
  temp_df <- rcar_ldas_disc %>%
    filter(Run == rcar_ldas_disc[["Run"]][i])
  temp_df_index <- (i - 1) %% points_per_run + 1
  if (temp_df_index > 1) {
    rcar_ldas_disc[i, "Disc"] <-
      estimate_discrepancy_CV(temp_df[1:temp_df_index,
                                      1:(96*96*3+3),
                                      drop = FALSE])
    # Error catch
    if( is.nan(as.numeric(rcar_ldas_disc[i, "Disc"]))) {
      print(i)
    }
  }
}

rcar_ldas_disc <- rcar_ldas_disc[ , (96*96*3 + 3 + 1):ncol(rcar_ldas_disc)]
gc()

# RCar OU ----
rcar_ou_disc <- read_csv("rcar_ou_disc.csv", col_names = FALSE)
names(rcar_ou_disc) <- c(paste0("Pixel", 1:(96*96*3)),
                           "Steer", "Gas", "Brake",
                           "Run", "Step", "Time_Elapsed")
rcar_ou_disc <- rcar_ou_disc %>%
  mutate(Method = "ou", Disc = NA) %>%
  arrange(Run)

# Discrepancy calculation
points_per_run <- rcar_ou_disc %>%
  filter(Run == 1) %>%
  nrow

for (i in 1:nrow(rcar_ou_disc)) {
  print(i / nrow(rcar_ou_disc) * 100)
  temp_df <- rcar_ou_disc %>%
    filter(Run == rcar_ou_disc[["Run"]][i])
  temp_df_index <- (i - 1) %% points_per_run + 1
  if (temp_df_index > 1) {
    rcar_ou_disc[i, "Disc"] <-
      estimate_discrepancy_CV(temp_df[1:temp_df_index,
                                      1:(96*96*3+3),
                                      drop = FALSE])
    # Error catch
    if( is.nan(as.numeric(rcar_ou_disc[i, "Disc"]))) {
      print(i)
    }
  }
}

rcar_ou_disc <- rcar_ou_disc[ , (96*96*3 + 3 + 1):ncol(rcar_ou_disc)]
gc()

# Combine rcar dataframes
rcar_disc <- rbind(rcar_unif_disc, rcar_ldas_disc, rcar_ou_disc)

# Export ----
write.csv(mcar_disc, "data/mcar_disc.csv")
write.csv(lunar_disc, "data/lunar_disc.csv")
write.csv(rcar_disc, "data/rcar_disc.csv")

# Import
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
    )
)

# Visualize ----
pdf("disc-results.pdf", width = 12, height = 5)
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
