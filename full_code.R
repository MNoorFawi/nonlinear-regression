library(ggplot2)
set.seed(13)
x <- rnorm(1000)
noise <- rnorm(1000, sd = 1.5)
y <- 4 * sin(3 * x) + cos(0.8 * x) - 1.5 * (x ^ 2 ) + noise
select <- runif(1000)
d <- data.frame(x = x, y = y)
training <- d[select > 0.1, ]
test <-d[select <= 0.1, ]

ggplot(training, aes(x = x, y = y)) + geom_point(alpha = 0.4) +
  geom_smooth(se = FALSE, method = 'lm', col = 'black')
linear_model <- lm(y ~ x, data = training)
resid_linear <- training$y - predict(linear_model)
## RMSE
sqrt(mean(resid_linear ^ 2))

library(mgcv)
gam_model <- gam(y ~ s(x), data = training)
gam_model$converged
resid_gam <- training$y - predict(gam_model)
## RMSE
sqrt(mean(resid_gam ^ 2))

library(e1071)
tuned <- tune.svm(y ~ x, training, kernel = 'radial',
                  gamma = c(seq(0.1, 1, 0.1), 10^(-6:-1)),
                  cost = 10^(-3:1))
tuned

svr_model <- svm(y ~ x, data = training, kernel = 'radial',
                 cost = 10, gamma = 0.5)
resid_svr <- training$y - predict(svr_model)
sqrt(mean(resid_svr ^ 2))

## test residuals
test_resid_svr <- test$y - predict(svr_model, test)
test_resid_gam <- test$y - predict(gam_model, test)
## test RMSE
sqrt(mean(test_resid_svr ^ 2))
sqrt(mean(test_resid_gam ^ 2))
## R Squared
cor(test$y, predict(gam_model, test)) ^ 2
cor(test$y, predict(svr_model, test)) ^ 2

training$svr_pred <- predict(svr_model, training)
training$gam_pred <- predict(gam_model, training)
ggplot(training, aes(x = x, y = y)) + 
  geom_point(alpha = 0.3) + 
  geom_line(aes(y = gam_pred, 
                color = 'GAM'), size = 1.3) +
  geom_line(aes(y = svr_pred, 
                color = 'SVR_radial'), size = 1.3) +
  scale_colour_manual(
    name="MODELS", 
    values=c(GAM = "#FF9999", SVR_radial = "#0072B2")) +
  theme(legend.position = c(0.8, 0.8)) +
  theme(legend.background=element_rect(fill="white", colour="black"))

