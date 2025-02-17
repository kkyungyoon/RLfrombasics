import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10 # 새로운 하이퍼파라미터(몇 틱의 데이터를 쌓아서 업데이트를 할지)
"""
value network가 평가를 대신해주기 때문에, 액터크리틱 방법론은 학습할 때 리턴이 필요없음
리턴 기다릴 필요없이 하나의 데이터가 생기면 네트워크 바로 업데이트 가능
"""

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    # policy network
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x)) # value network와 하나의 레이어 공유
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    # value network : 학습에만 사용되기때문에 main에서 호출될 일 없음
    def v(self, x):
        x = F.relu(self.fc1(x)) # policy network와 하나의 레이어 공유
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        """
        n_rollout동안 모인 데이터를 s끼리, a끼리, r끼리, s'끼리, done끼리 모아서 미니배치 만들기
        """
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        """
        policy network의 loss
        -torch.log(pi_a) * delta.detach()
        detach 함수 : delta를 상수취급하기위해(해당 값을 계산하기까지 필요했던 모든 그래프 연산들을 떼어냄)

        value network의 loss 
        TD 방식을 이용해 계산
        td_target.detach(): 정답은 그 자리에 가만히 있게하려고 상수취급

        """
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
      
def main():  
    env = gym.make('CartPole-v1')
    model = ActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float()) # 액션을 뽑고
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a) # 액션을 환경에 던지고, 상태전이와 보상관찰
                model.put_data((s,a,r,s_prime,done)) # 저장
                
                s = s_prime
                score += r
                
                if done:
                    break                     
            
            model.train_net() # 학습진행
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()