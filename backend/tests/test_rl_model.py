from app.rl_model import train_rl_model, predict

def test_predict():
    model = train_rl_model()
    state = [0, 0, 0, 0]
    action = predict(model, state)
    assert isinstance(action, int)