"""
MDP : unknown = model-free
prediction 방법 : MC, TD
그 중 TD prediction
- MC와 environment, agent는 동일
- main함수 : 
    1) 에피소드가 끝나고 업데이트하는게 아니라, 한번의 액션마다 테이블이 업데이트됨
    2) 업데이트 폭인 alpha가 MC보다 크다.(왜냐, TD가 MC에 비해 학습의 변동성이 작은 덕분에 그만큼 큰 폭의 업데이트 가능)
"""
import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    
    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a==0:
            self.move_left()
        elif a==1:
            self.move_up()
        elif a==2:
            self.move_right()
        elif a==3:
            self.move_down()

        reward = -1 # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        self.y += 1  
        if self.y > 3:
            self.y = 3
      
    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0
      
    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
  
    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else :
            return False

    def get_state(self):
        return (self.x, self.y)
      
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent():
    def __init__(self):
        pass        

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action


def main():
    #TD
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.01

    # 50000 에피소드
    for k in range(50000):
        done = False
        # 에피소드가 끝날 때까지 반복
        while not done:
            # 현재 state 저장
            x, y = env.get_state()
            action = agent.select_action()
            (x_prime, y_prime), reward, done = env.step(action)
            # 업데이트된 state 저장
            x_prime, y_prime = env.get_state()
            # TD(0) 업데이트 식 : V(s) ← V(s) + α * (R + γ * V(s') - V(s))
            data[x][y] = data[x][y] + alpha*(reward+gamma*data[x_prime][y_prime]-data[x][y])
        # 에피소드 종료 후 환경 초기화
        env.reset()
            
    for row in data:
        print(row)

if __name__ == '__main__':
    main()