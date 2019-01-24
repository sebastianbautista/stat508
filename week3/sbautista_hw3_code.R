part1 = function() {  
  library(Matrix)
  
  generate.y = function(copies, X, b) { # generate multiple copies of the vector y and store in a list
    lapply(1:copies, function(i) X%*%b + rnorm(nrow(X),0,1))
  } #generate.y
  
  beta.hat = function(sam=c(1:nrow(X)), A, X, y.vec) { # least-squares estimate of beta (vector), stored in a list
    b = vector("list", length(A)) # initialize list of same length as length of A
    for (i in 1:length(A)) { # using subset of X corresponding to columns in A[[i]] and observations in 'sam'
      b[[i]] = solve( t(X[sam, A[[i]]]) %*% X[sam, A[[i]]] ) %*% t(X[sam, A[[i]]]) %*% y.vec 
    }
    return(b)
  } # beta.hat
  
  get.ASPE = function(sam=c(1:nrow(X)), A, X, z, w) {
    PE = vector("list", length(z)) # initialize list of lists
    for (k in 1:length(z)) {
      PE[[k]] = vector("list", length(A))
    }
    for (i in 1:length(A)) {
      if (length(X[-sam, A[[i]]]) > 0) {
        for (k in 1:length(z)) {
          PE[[k]][[i]] = (1/length(z[[k]])) * sum( ( z[[k]] - X[-sam, A[[i]]] %*% w[[k]][[i]] )^2)
        }
      }
    }
    return(PE)
  } # get.ASPE
  
  optimal.models = function(A, X, y) {
    
    # Split the data using 1/3 for training and 2/3 for testing
    sam = sample(c(1:nrow(X)), round(nrow(X)/3, 0))
    ym = vector("list", length(y))
    yl = vector("list", length(y))
    for (i in 1:length(y)) {
      ym[[i]] = y[[i]][sam]
      yl[[i]] = y[[i]][-sam]
    }
    
    # Compute least-squares estimate using only the training set
    b = vector("list", length(ym))
    for (i in 1:length(ym)) {
      b[[i]] = beta.hat(sam, A, X, ym[[i]]) 
    }
    
    # Evaluating average squared prediction error using the testing part of the data
    ASPE = get.ASPE(sam, A, X, yl, b)
    
    # Pick out the model with the smallest average squared prediction error
    optimal.val = sapply(lapply(ASPE, function(x) sort(unlist(x))), function(x) x[1])
    optimal.mod = vector("character",length(optimal.val))
    for (i in 1:length(optimal.val)) {
      tmp = names(A[abs(unlist(ASPE[[i]]) - optimal.val[i]) < 0.0001]) # equal to smallest ASPE to within 0.0001 precision (can be modified)
      if (length(tmp) > 1) { # in case of (near) ties, break tie at random
        optimal.mod[i] = sample(tmp, 1)
      } else{
        optimal.mod[i] = tmp
      }
    }
    
    # tabulate the findings, including any models for which the frequency of selection is zero
    freq = vector("integer", length(A))
    names(freq) = names(A)
    freq[names(table(optimal.mod))] = table(optimal.mod)
    
    # return the frequency table as well as the subset of observations selected for the training set
    return(list(freq=freq, train=sort(sam))) 
  } # optimal.models
  
  ######################
  obs = 100
  
  ## Create collection of models
  x1 = rep(1,obs)
  x2 = 5*(1:obs) / obs
  x3 = x2*sqrt(x2)
  x4 = x2^2
  x5 = log(x2)
  x6 = x2*x5
  X = cbind(x1,x2,x3,x4,x5,x6)
  dimnames(X)[[1]] = c(1:obs)
  
  rm(x1,x2,x3,x4,x5,x6,obs) # clean up: variables only needed to build the input matrix X
  
  A = list(c(1,2,3,5,6), c(1,2,4,5,6), 
           c(1,2,5,6), c(1,3,5,6), c(1,4,5,6),c(2,3,5,6), c(2,4,5,6), 
           c(1,2,3), c(1,2,4), c(1,2,5), c(1,2,6), c(4,5,6),
           c(1,2), c(1,3), c(1,5), c(4,5), c(2,4))
  names(A) = c("(1,2,3,5,6)", "(1,2,4,5,6)", 
               "(1,2,5,6)", "(1,3,5,6)", "(1,4,5,6)", "(2,3,5,6)", "(2,4,5,6)", 
               "(1,2,3)", "(1,2,4)", "(1,2,5)", "(1,2,6)", "(4,5,6)", 
               "(1,2)", "(1,3)", "(1,5)", "(4,5)", "(2,4)") 
  
  ## Implement simulations ##
  
  ## Simulation 1: Distribution of optimal models for fixed split over different instances of y
  copies = 1000
  
  set.seed(17)
  ylist = generate.y(copies, X[, A[[9]]], rep(2,3))
  
  set.seed(11)
  seed.optimal = optimal.models(A, X, ylist)
  seed.optimal
  plot(seed.optimal$freq, type="h", main=paste("Proportion of times each model is selected as the optimal model\n for an arbitrary but fixed split over", copies, "instances of y"), xlab="", ylab="", bty="n", axes=F)
  axis(1, at=1:length(A), labels=names(A), las=2)
  points(1:length(A), seed.optimal$freq, pch=20)
  text(x=1:length(A), y=seed.optimal$freq+copies/100, labels=round(seed.optimal$freq/copies, 2), cex=0.7)
  
  ## Simulation 2: Distribution of optimal models for single instance of y over multiple splits
  splits = 1000
  
  set.seed(17)
  ylist = generate.y(1, X[, A[[9]]], rep(2,3))
  
  set.seed(11)
  iter.optimal = vector("list", splits)
  for (iter in 1:splits) {
    iter.optimal[[iter]] = optimal.models(A, X, ylist)
  }
  
  ## Sum across iterations to get frequency of selection of each model
  iter.freq = iter.optimal[[1]]$freq
  for (iter in 2:splits) {
    iter.freq = iter.freq + iter.optimal[[iter]]$freq
  }
  
  iter.freq
  
  plot(iter.freq, type="h", main=paste("Proportion of times each model is selected as the optimal model\n for a single instance of y over", splits, "splits"), xlab="", ylab="", bty="n", axes=F)
  axis(1, at=1:length(A), labels=names(A), las=2)
  points(1:length(A), iter.freq, pch=20)
  text(x=1:length(A), y=iter.freq+splits/100, labels=round(iter.freq/splits, 2), cex=0.7)

  get.Cp = function(A, X, z, w) {
    Cp = vector("list", length(z)) # initialize list of lists
    for (k in 1:length(z)) {
      Cp[[k]] = vector("list", length(A))
    }
    
    for (i in 1:length(A)) {
      for (k in 1:length(z)) {
        # get estimate of variance from largest model that (you hope) contains the true model
        sigma2 = summary(lm(z[[k]] ~ -1 + X[,1] + X[,2] + X[,3] + X[,4] + X[,5] + X[,6]))$sigma^2
        # compute Cp using formula (6.2) in ISLR Section 6.1.3
        Cp[[k]][[i]] = (1/nrow(X))*(sum((z[[k]] - X[, A[[i]]] %*% w[[k]][[i]])^2) + 2*length(A[[i]])*sigma2)
      }
    }
    return(Cp)
  } # get.Cp
  
  optimal.models.Cp = function(A, X, y) {
    
    # Compute least-squares estimates using full data
    b = vector("list", length(y))
    for (i in 1:length(y)) {
      b[[i]] = beta.hat(A=A, X=X, y.vec=y[[i]]) 
    }
    
    # Evaluate Mallows Cp
    Cp = get.Cp(A, X, y, b)
    
    # Pick out the model with the smallest value
    optimal.val = sapply(lapply(Cp, function(x) sort(unlist(x))), function(x) x[1])
    optimal.mod = vector("character",length(optimal.val))
    for (i in 1:length(optimal.val)) {
      tmp = names(A[abs(unlist(Cp[[i]]) - optimal.val[i]) < 0.0001]) # equal to smallest Cp to within 0.0001 precision (can be modified)
      if (length(tmp) > 1) { # in case of (near) ties, break tie at random
        optimal.mod[i] = sample(tmp, 1)
      } else{
        optimal.mod[i] = tmp
      }
    }
    
    # tabulate the findings, including any models for which the frequency of selection is zero
    freq = vector("integer", length(A))
    names(freq) = names(A)
    freq[names(table(optimal.mod))] = table(optimal.mod)
    
    # return the frequency table as well as the subset of observations selected for the training set
    return(list(freq=freq)) 
  } # optimal.models.Cp
  
  Cp.optimal = optimal.models.Cp(A, X, ylist)
  print(Cp.optimal)
}