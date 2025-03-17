# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

![nn](https://github.com/user-attachments/assets/7661c962-df9a-4a31-a2c0-c78008678925)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Siva Chandran R
### Register Number: 212222240099
```python
# Name: Siva Chandran R
# Register Number: 212222240099
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 5)
    self.fc2 = nn.Linear(5, 7)
    self.fc3 = nn.Linear(7, 1)
    self.relu = nn.ReLU()
    self.history = {'loss':[]}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

![image](https://github.com/user-attachments/assets/0a1b7c2b-b4ab-41f2-8a06-7837448ca3c1)



## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-17 112423](https://github.com/user-attachments/assets/99e1e326-9a75-471e-8b72-15c4f1b9f50f)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/87cc534d-b842-497c-b69a-57deb5359b6c)



## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
