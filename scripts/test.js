//激活函数
function Relu(z) {
    let res = (z>0)? z:0;
    return res;
}
//激活函数求导
function Relu_f(z) {
    return (z>0)?1:0;
}
//随机数生成
function random(min, max) {
    const num = Math.floor(Math.random() * (max - min)) + min;
    return num;
}
function randomfloat(min, max) {
    const num = Math.random() * (max - min) + min;
    return num;
}
//求和函数
function addinput(input) {
    var sum = 0.0;
    for (var i = 0; i < input.length; i++) {
        sum += input[i];
    }
    return sum;
}
function addinputweight(input, weight, b) {
    var sum = b;
    for (var i = 0; i < input.length; i++) {
        sum += input[i] * weight[i];
    }
    return sum;//激活层未处理
}
//验证集生成
function getInputSet(m) {
    let input = Array.from({ length: m });
    for(var i = 0 ;i < m ; i++)
    {
        input[i] = randomfloat(0.0, 10.0);
    }
    return input;
}
//前向传播
function forward(sum) {
    return Relu(sum);//激活层处理
}
//反向传播
function backward(input, output, loss, weights) {
    var grad = Array.from({ length: weights.length });
    for (var j = 0; j < weights.length; j++) {
        grad[j] = (loss >= 0) ? (input[j] * Relu_f(output)): (-input[j] * Relu_f(output));
    }
    var grad_b = (loss >= 0)?Relu_f(output):-Relu_f(output);
    var obj = [grad, grad_b];
    //console.log(obj);
    return obj;
}
//优化参数
function optimize(weights,b,grad,grad_b,learning_rate)
{
    for(var i = 0; i < weights.length; i++)
    {
        weights[i] -= grad[i] * learning_rate;//优化weight
    }
    b -= grad_b*learning_rate;//优化b
    var obj = [weights,b];
    return obj;
}
function train(weights, train_steps, learning_rate, m, b) {
    let grad = Array.from({ length: m });//梯度
    let train_output;
    let origin_loss = Array.from({ length: train_steps });
    let abs_loss = Array.from({ length: train_steps });
    let result;
    let para;
    for (var i = 0; i < train_steps; i++) {
        //生成训练样本
        train_input = getInputSet(m);
        output = addinputweight(train_input, weights, b);//未通过激活层
        train_output = forward(addinputweight(train_input, weights, b));//前向传播
        var answer = addinput(train_input);//模拟求和函数

        origin_loss[i] = train_output - answer;//带符号的loss
        abs_loss[i] = Math.abs(origin_loss[i]);//L1 Loss

        /* console.log('第' + (i + 1).toString() + '次训练前的weights are [' + weights.toString() + ']');
        console.log('第' + (i + 1).toString() + '次训练前的b is ' + b.toString());
        console.log('第' + (i + 1).toString() + '次训练前的loss为' + origin_loss[i] + ',loss rate为' + Math.abs(abs_loss[i]) / answer);
         *///反向传播
        result = backward(train_input, train_output, origin_loss[i], weights);
        grad = result[0];
        grad_b = result[1];
        //weights[j] -= grad[j] * learning_rate;//优化
        para = optimize(weights,b,grad,grad_b,learning_rate);
        weights = para[0];
        b = para[1];
    }
    console.log('训练完成，loss:['+abs_loss+']');
    return para;
}
let m = 10;
let input = Array.from({ length: m });
let weights = Array.from({ length: m });
let b = randomfloat(0.0, 10.0);
let output;
//y = W1X1+W2X2+W3X3+...+W10X10+b;
//随机初始化输入向量
input = getInputSet(m);
//输出向量维数
console.log('input dimension is ' + m.toString());
//显示输入向量
console.log('verify input is [' + input.toString() + ']');
//随机初始化参数向量
for (var i = 0; i < m; i++) {
    weights[i] = randomfloat(-3.0, 3.0);
}
//显示参数初始化结果
console.log('weights are [' + weights.toString()+','+b.toString()+ ']');
let answer = addinput(input);
//输出正确结果
console.log('the sum of the input is ' + answer);
//单层神经网络训练
//训练参数设置
train_steps = 600;//训练轮数
learning_rate = 0.03;//学习率
let train_input = [];
let para;
para = train(weights, train_steps, learning_rate, m, b);//训练
weights = para[0];
b = para[1];
console.log('训练后的weights为[' + weights.toString() +','+b.toString()+ ']');
let final_answer = addinputweight(input, weights,b);
console.log('计算input:[' + input + ']训练后拟合求到的答案为：' + final_answer + ',实际计算结果为：' + answer + ',loss_rate为：' + Math.abs(final_answer - answer) / answer);

