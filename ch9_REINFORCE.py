import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128) # 입력 크기 : CartPole 환경의 상태
        self.fc2 = nn.Linear(128, 2) # 출력 크기 : CartPole의 행동(왼쪽, 오른쪽)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
    # train을 위해 data 저장 : (reward, 선택한 행동의 확률)
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        """
        실제로 네트워크를 학습하는 코드
        """
        R = 0
        self.optimizer.zero_grad() # 기존 gradient 초기화
        for r, prob in self.data[::-1]:
            R = r + gamma * R # 누적 보상
            loss = -torch.log(prob) * R # Policy Gradient loss function
            loss.backward() # loss에 대한 gradient가 계산되어 계속 더해짐
        self.optimizer.step() # 축적된 gradient로 파라미터 업데이트
        self.data = [] # 학습 후 데이터 비우기

def main():
    env = gym.make('CartPole-v1')
    pi = Policy() # policy network
    score = 0.0
    print_interval = 20
    
    for n_epi in range(10000):
        s, _ = env.reset() # 환경 초기화
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float()) # 액션별 확률
            m = Categorical(prob) # 확률분포 객체 생성(torch.distributions.Categorical 클래스의 객체를 생성)
            a = m.sample() # 확률에 따라 액션 샘플링(확률이 높은 액션은 더 자주, 확률이 낮은 액션은 덜 뽑히게 됨)
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r,prob[a])) # loss계산을 위해서
            s = s_prime # 상태 업데이트
            score += r
            
        pi.train_net() # 네트워크 학습
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0 # 점수 초기화
    env.close()
    
if __name__ == '__main__':
    main()