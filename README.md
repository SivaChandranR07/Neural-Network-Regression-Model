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
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

![Screenshot 2025-03-05 213650](https://github.com/user-attachments/assets/221b58ee-754c-49b0-aec0-18a586593417)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-05 213909](https://github.com/user-attachments/assets/5af73e9b-37ce-45c6-a801-854ccb6f45c8)


### New Sample Data Prediction

![Screenshot 2025-03-05 213921](https://github.com/user-attachments/assets/23fbacb0-b7d5-4791-b60f-41cd9b1cb0c6)


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
