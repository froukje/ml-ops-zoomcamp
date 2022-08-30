import predict

input_example = {
    "X1": 0.98,
    "X2": 514.50,
    "X3": 294.00,
    "X4": 110.25,
    "X5": 7.00,
    "X6": 2,
    "X7": 0.00,
    "X8": 0,
}

features = predict.preprocess(input_example)
pred = predict.predict(features)

print(pred)
