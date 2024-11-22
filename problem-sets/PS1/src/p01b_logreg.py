import numpy as np
import util 

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    print('now main is running!')
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.which
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval,y_eval=util.load_dataset(eval_path,add_intercept=True)
    newtown=LogisticRegression()
    #print(x_train.shape)
    newtown.fit(x_train,y_train)
    print(newtown.theta)
    #newtown.theta=np.array([-6.26018491  ,2.47707251,-0.0299125])加不加结果一样，说明可以了
    pred=newtown.predict(x_eval)
    #print('now reday to plot')
   # util.plot(x_train,y_train,newtown.theta)
   # util.plot(x_eval,pred,newtown.theta).    plot模块还是有问题
   # print('now plot done')
    print(np.mean(y_train==newtown.predict(x_train)))
    print(np.mean(pred==y_eval))



class LogisticRegression(LinearModel):
   
    def fit(self, x, y):
        m,n = x.shape
        self.theta = np.array([0.0,0.0,0.0])#只能全0，不然为奇异矩阵，H对初始点很敏感
    # Newton's method
        min=10
        i=0
        while i<min:
        # Save old theta
            theta_old = np.copy(self.theta)
            print('count')#4 times to converge
        # Compute Hessian Matrix
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            H = (x.T* h_x * (1 - h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x - y) / m
        # Updata theta
            self.theta -=np.linalg.inv(H).dot(gradient_J_theta)
            i+=1
        # End training
            
    # *** END CODE HERE ***
    def predict(self, x):
    
        return (1 / (1 + np.exp(-x.dot(self.theta))))>=0.5

main(train_path='mnt\\d\\ml\\problem-sets\\PS1\\data\\ds1_train.csv',
      eval_path='mnt\\d\\ml\\problem-sets\\PS1\\data\\ds1_valid.csv',
      pred_path='output/p01b_pred_1.txt')