# Biometric-bike-analyzing
Biometrics Analysis: 
# brake
# pedal
# cadence
# phone (accelerometer)

Session(Day) = 1,2
Track = A(large), B(small)

Results:
1. Identification Performance (IP)
--- On each sensor individually 
--- Combinations of sensors (pairs)
--- All 


2. IP:
--- Session 1 
------- Small only: cross validation 
1,2,3,4,5 
train 1,2,3,4 test 5
train 2,3,4,5 test 1
train 3,4,5,1 test 2 
and so on
--- Session 2 
--- Train Session 1, Test Session 2
---


Preprocessing
- remove noise 
- segment the repetitions of the small track (hint: an example location)
- %remove parts where not all participants cycled 

ML Model (RF, SVM) 
train(x, label) test(y): output is label, accuracy (F1-score)

brake 
*** Day 1, Track B, train(BrakeData(time and value), PID), Rep 1-5 small track
80-20
1,2,3,4,5 
train 1,2,3,4 test 5 - output 1
train 2,3,4,5 test 1 - output 2
train 3,4,5,1 test 2 - output 3 
....
....

average: cross-validation - output 6

*** Day 2, Track B, train(BrakeData(time and value), PID),
same procedure, only data from day 2 

*** BOTH: 
Train with all data from day 1, test with all data from day 2 - output 7

Train with all data from day 2, test with all data from day 1 - output 8 

Average of both sessions - output 9


*** Large Track ONLY: Day 1, Track A - Training, Day 2, Track A - Testing

Vice versa 



**** train with small track and test with large track 
train with the repetitions (1-5) test with large track 

day 1 
day 2
average both days 



Key Differences
Aspect	            Current Approach	                    F1-Score Approach (Supervised)
Type of learning	Unsupervised (novelty detection)	    Supervised (binary or multiclass)
Training data	    Only laps 1–4 from one rider	        Data from multiple riders, labeled
Test data	        Lap 5 (no label, just “Does it fit?”)	Labeled samples from same & other riders
Goal	            Detect if test is “same or different”	Predict labels correctly
Metrics used	    Inlier ratio (just % of matching)	    F1, Precision, Recall, Accuracy
Model type	        One-Class SVM / IsolationForest	        SVM, RandomForest, XGBoost, etc.
Label required?	    No	                                    Yes
                                                            Average of all F1 scores to know how good it is 
                                                            Confusion matrix 


                                                            TODO
                                                            Print the label of each values used
                                                            Explain the confusion 
                                                            Try non liniar SVM 
                                                            what happens when we train with the 5 laps from day 1 and test with 5 laps of day 2


                                                            SVM (linear vs polynomial), SVM RBF kernel
                                                            KNN K-Nearest Neighbor
                                                            Precision and recall results please
                                                            feature importance as output

                                                            Visual inspection of the data to kn
