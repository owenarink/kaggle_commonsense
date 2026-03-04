from models import mlp_tfidf
from features import tfidf
import preprocessing
from torch.utils.data import DataLoader, TensorDataset


def load_prep_data():
    x_train_raw = pd.read_csv('data/train_data.csv')
    x_test_raw = pd.read_csv('data/test_data.csv') 
    y_train_raw = pd.read_csv('data/train_answers.csv')
    
    x_train_preprocessed, y_train_preprocessed = preprocess(x_train_raw, x_test_raw, y_train_raw)
    x_train, x_val, y_train, y_val = get_tfidf(x_train_preprocessed, y_train_preprocessed)

    return x_train, x_val, y_train, y_val

def train():
    train_dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
    val_dataset = TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).float())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model
    model = MLP(n_inputs=x_train.shape[1], n_hidden=64, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_crit = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            #forward
            outputs = model(batch_x)
            loss = loss_crit(outputs, batch_y)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    load_prep_data()
    train()