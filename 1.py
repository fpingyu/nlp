import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate
#设置相同的参数用于比较三个模型的效果
#便于复现
seed = 1234
#NumPy库随机数种子设置
random.seed(seed)
np.random.seed(seed)
#PyTorch库的随机数种子设置为指定的seed值
torch.manual_seed(seed)
#将PyTorch库的CUDA后端的随机数种子设置为指定的seed值
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
#加载一个名为"multi30k"的英文-德文翻译数据集
dataset = datasets.load_dataset("multi30k")
#数据集结构
dataset
#这是一个简单的字典操作
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)
train_data[0]
#一个小型版本的英、德语模型，它经过了深度学习训练，可以处理常见的英语文本。这种模型对于处理大量文本数据非常有用，因为它可以有效地利用硬件
#"good morning!"转换为数组["good", "morning", "!"]
en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")
# 列表推导式生成['What', 'a', 'lovely', 'day', 'it', 'is', 'today', '!']
string = "What a lovely day it is today!"

[token.text for token in en_nlp.tokenizer(string)]
'''
这个函数的目的是将一个英文和德文的例子字符串分别转换为token（词汇）列表，并确保每个列表的长度都不超过指定的最大长度。
同时，函数还提供了将token转换为小写的选项，以及为每个token列表添加起始符（<SOS>）和结束符（<EOS>）的功能。
'''
def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}
'''
这段代码是用于定义一些参数和配置，然后使用这些参数和配置对训练数据、验证数据和测试数据进行映射（map）。
具体来说，这段代码的作用是将文本数据tokenize，以便后续进行训练、验证和测试.
'''
max_length = 1_000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

fn_kwargs = {
    "en_nlp": en_nlp,
    "de_nlp": de_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}
'''ss
结构为
Dataset({
    features: ['en', 'de', 'en_tokens', 'de_tokens'],
    num_rows: 29000
})
'''
train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)
train_data[0]
'''
使用PyTorch和torchtext库构建英语和德语词汇表（vocab）
'''
#表示最小频率阈值，只有出现至少两次的单词才会被添加到词汇表中
min_freq = 2
#表示未知字符，当输入的数据中包含未在词汇表中出现的字符时，将使用这个字符作为替换
unk_token = "<unk>"
#表示填充字符，当需要将输入数据pad成相同的长度时使用
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]
#en_vocab：是一个词汇表，用于存储英语词汇,
# build_vocab_from_iterator函数会遍历训练数据中的英语词汇，并根据最小频率阈值和特殊字符来构建词汇表
en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)
# de_vocab：是一个词汇表，用于存储德语词汇
de_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["de_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)
en_vocab.get_itos()[:10]
en_vocab.get_itos()[9]
de_vocab.get_itos()[:10]
en_vocab.get_stoi()["the"]
en_vocab["the"]
len(en_vocab), len(de_vocab)
"the" in en_vocab
"The" in en_vocab
'''
这段Python代码的含义是对英语词汇表和德语词汇表进行校验，确保它们具有相同的未知token（unk_token）和填充token（pad_token)
'''
assert en_vocab[unk_token] == de_vocab[unk_token]
assert en_vocab[pad_token] == de_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]
#我们可以设置当我们尝试获取词汇表之外的标记的索引时返回的值。在本例中，为未知令牌的索引<unk>。
en_vocab.set_default_index(unk_index)
de_vocab.set_default_index(unk_index)
en_vocab["The"]
en_vocab.get_itos()[0]
tokens = ["i", "love", "watching", "crime", "shows"]
en_vocab.lookup_indices(tokens)
en_vocab.lookup_tokens(en_vocab.lookup_indices(tokens))
'''
是将example中的英语和德语tokens转换为对应的索引列表
实现原理是将英文和德文句子分别转换为词汇表中的索引列表，然后将这两个列表作为字典的键值对返回。
'''
def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}
fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)
train_data[0]
en_vocab.lookup_tokens(train_data[0]["en_ids"])
'''
数据集转换为指定的数据类型（PyTorch张量）并按照指定的列格式进行格式化
'''
data_type = "torch"
format_columns = ["en_ids", "de_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)
train_data[0]
type(train_data[0]["en_ids"])
'''
数据加载
'''
#数用于对数据批进行处理，以便在训练或验证时传入给模型
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_de_ids = [example["de_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch

    return collate_fn
#创建一个数据加载器，它用于从给定的数据集（dataset）中加载批量数据
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader
#这段代码主要设置了三个数据加载器：训练数据加载器、验证数据加载器和测试数据加载器
#批次大小，即每批数据的大小
batch_size = 128

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)
'''
这段代码是定义一个编码器的Python类，用于PyTorch深度学习框架
定义了一个基于LSTM（Long Short-Term Memory）的编码器，它可以处理序列数据
'''
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
         # 设置隐藏层的维度
        self.hidden_dim = hidden_dim
        # 是RNN中的层数
        self.n_layers = n_layers
        #嵌入层、LSTM层和Dropout层
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell
    '''
    这段代码是定义一个解码器类（Decoder）
    '''
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        #设置解码器的输出维度
        self.output_dim = output_dim
        # 设置隐藏层的维度
        self.hidden_dim = hidden_dim
        # 是RNN中的维度
        self.n_layers = n_layers
        # 设置嵌入层的维度
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        # 设置LSTM层
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        #全连接层
        self.fc_out = nn.Linear(hidden_dim, output_dim)
         # 设置Dropout层
        self.dropout = nn.Dropout(dropout)
         #定义前向传播过程
    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
'''
是一个用于序列到序列学习的PyTorch模型。
序列到序列学习是一种用于语言模型或文本生成任务的算法，它将输入序列映射到输出序列。
在这个模型中，我们使用编码器-解码器结构，其中编码器用于处理输入序列，解码器用于生成输出序列。
'''
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
'''
在这个模型中，我们将德语的词汇表（de_vocab）映射到英语的词汇表（en_vocab），同时将德语句子转换为英语句子
'''

input_dim = len(de_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)
'''
这段代码是用于初始化PyTorch模型权重的一个函数。
它的作用是遍历模型中的所有参数（包括卷积层、全连接层等），并将这些参数的值随机初始化为一个在[-0.08, 0.08]范围内的均匀分布。
'''
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
'''
这段代码是一个Python函数，用于计算一个模型中的可训练参数数量,1389万参数量
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")
'''
我们定义优化器，我们使用它来更新训练循环中的参数
'''
optimizer = optim.Adam(model.parameters())
'''
损失函数
'''
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
'''
定义训练并返回所有批次的平均损失
'''
def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
'''
定义评估函数
'''
def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
'''
训练
'''
n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")
'''
首先要做的是在测试集上测试模型的性能
'''
model.load_state_dict(torch.load("tut1-model.pt"))

test_loss = evaluate_fn(model, test_data_loader, criterion, device)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")
'''
首先要做的是在测试集上测试模型的性能
'''
model.load_state_dict(torch.load("tut1-model.pt"))

test_loss = evaluate_fn(model, test_data_loader, criterion, device)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")
'''
该函数使用深度学习模型（model）将输入的德语句子（sentence）翻译为英语句子
'''
def translate_sentence(
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = de_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens
'''
传递一个测试用例(模型尚未训练过的内容)作为句子来测试翻译语句功能，传入德语句子并期望得到类似英语句子的内容
'''

#测试集对应的德-英
sentence = test_data[0]["de"]
expected_translation = test_data[0]["en"]

sentence, expected_translation
translation = translate_sentence(
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
)
translation
sentence = "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt."
translation = translate_sentence(
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
)
#模型测试结果
# 我们收到了我们的翻译，效果还可以。
translation
'''
现在，遍历测试数据，获得模型对每个测试句子的翻译。
'''
translations = [
    translate_sentence(
        example["de"],
        model,
        en_nlp,
        de_nlp,
        en_vocab,
        de_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )
    for example in tqdm.tqdm(test_data)
]
'''
BLEU指标可以从评估库中加载
'''
bleu = evaluate.load("bleu")

predictions = [" ".join(translation[1:-1]) for translation in translations]

references = [[example["en"]] for example in test_data]
predictions[0], references[0]
'''
大写转小写
'''
def get_tokenizer_fn(nlp, lower):
    def tokenizer_fn(s):
        tokens = [token.text for token in nlp.tokenizer(s)]
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens

    return tokenizer_fn
tokenizer_fn = get_tokenizer_fn(en_nlp, lower)
tokenizer_fn(predictions[0]), tokenizer_fn(references[0][0])
'''
计算测试集中的BLEU度量
'''
results = bleu.compute(
    predictions=predictions, references=references, tokenizer=tokenizer_fn
)
results