const canvas = document.getElementById("canvas");
const rect = canvas.getBoundingClientRect()
const ctx = canvas.getContext("2d");
const trainingInput = document.getElementById("training");
const letterInput = document.getElementById("letter");
const delay = 50;
const clearButton = document.getElementById("clear");
const output = document.getElementById("out");

let drawing = false;
let interval = null;

let training = false;

trainingInput.onchange = e => {
    training = trainingInput.checked;
    output.innerText = training ? 'Training mode activated. Any letters drawn will be contributed to local model accuracy. Write carefully!' : 'Draw a letter and a let the model predict!';
}

let x, y;
let lastX, lastY;

let xVels = [];
let yVels = [];

let lettersData = window.localStorage.lettersData ? JSON.parse(window.localStorage.lettersData) : [];

clearButton.onclick = e => {
    window.localStorage.lettersData = "";
}

canvas.onmousedown = e => {
    x = e.clientX - rect.left;
    y = e.clientY - rect.top;
    lastX = x;
    lastY = y;
    xVels = [];
    yVels = [];
    ctx.fillStyle = "#FF0000";
    ctx.fillRect(x, y, 5, 5);
    drawing = true;
}

canvas.onmouseup = e => {
    drawing = false;
    lastX = lastY = null;
    if (training) {
        let letterIndex = lettersData.findIndex(l => l.letter == letterInput.value);
        let letter = lettersData[letterIndex];
        if (letter) {
            letter.initialData[0].push(xVels);
            letter.initialData[1].push(yVels);
        } else {
            letter = {
                letter: letterInput.value,
                stateCount: [3, 3],
                initialData: [[xVels], [yVels]]
            }
            lettersData.push(letter);
            letterIndex = lettersData.length - 1;
        }
        // trainAll();
        lettersData[letterIndex] = trainLetter(lettersData[letterIndex]);
        window.localStorage.lettersData = JSON.stringify(lettersData);
    }
    let prob = -1
    let letter;
    lettersData.forEach(data => {
        let pX = viterbi(xVels, data.stateCount[0], data.transitionProbs[0], data.emissionProbs[0])
        let pY =  viterbi(yVels, data.stateCount[1], data.transitionProbs[1], data.emissionProbs[1])
        let p = pX * pY;
        if (p > prob) {
            prob = p;
            letter = data.letter;
        }
    });
    output.innerText = training ?  `Training on ${letter}... uncheck box for prediction` : `Model predicts ${letter}`

    if (!training)
    if (interval) {
        clearInterval(interval);
        interval = null;
    }
    clearCanvas();
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

canvas.onmousemove = e => {
    if (drawing) {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
        ctx.fillStyle = "#FF0000";
        ctx.fillRect(x, y, 5, 5);

        if (!interval) {
            interval = setInterval(() => {
                if (lastX && lastY) {
                    let dx = (x - lastX) / delay;
                    let dy = -(y - lastY) / delay;

                    if (Math.abs(dx) >= 0.01 || Math.abs(dy) >= 0.01) {
                        xVels.push(dx);
                        yVels.push(dy);
                    }
                }
                lastX = x;
                lastY = y;
            }, delay);
        }
    }

}

function meanAndStd (array) {
    const n = array.length
    const mean = array.reduce((a, b) => a + b) / n
    const std = Math.sqrt(array.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n)
    return [mean, std]
  }

  function train(data, stateCount, cycles = 10) {
    data = data.map(d => {
        let avg = Math.floor(d.length / stateCount);
        let splitUp = [];
        while (d.length >= avg && splitUp.length < stateCount) {
            splitUp.push(d.splice(0, avg))
        }
        let origLeftover = d.length;
        while (d.length > 0) {
            splitUp[origLeftover - d.length].push(d.shift())
        }
        return splitUp;
    });

    curMs = []
    for (let k = 0; k < cycles; k++) {
        curMs = [];
        for (let i = 0; i < stateCount; i++) {
            let totalState = [];
            for (let j = 0; j < data.length; j++) {
                totalState.push(...data[j][i])
            }
            curMs.push(meanAndStd(totalState))
        }

        data = data.map(d => {
            for (let i = 0; i < stateCount; i++) {
                if (i != 0) {
                    while (true) {
                        if (d[i].length <= 1) {
                            break;
                        }
                        let thisRelation = Math.abs(d[i][0] - curMs[i][0]) / curMs[i][1];
                        let otherRelation = Math.abs(d[i][0] - curMs[i - 1][0]) / curMs[i - 1][1];

                        if (thisRelation > otherRelation) {
                           let removed = d[i].shift();
                           d[i - 1].push(removed); 
                        } else {
                            break;
                        }
                    }
                }

                if (i != stateCount - 1) {
                    while (true) {
                        if (d[i].length <= 1) {
                            break;
                        }
                        let thisRelation = Math.abs(d[i][d[i].length - 1] - curMs[i][0]) / curMs[i][1];
                        let otherRelation =  Math.abs(d[i][d[i].length - 1] - curMs[i + 1][0]) / curMs[i + 1][1];
                        
                        if (thisRelation > otherRelation) {
                           let removed = d[i].pop();
                           d[i + 1].unshift(removed); 
                        } else {
                            break;
                        }
                    }
                }
            }
            return d;
        });
    }

    let transitionProbs = [];
    for (let i = 0; i < stateCount; i++) {
        let sumLength = 0;
        data.forEach(d => {
            sumLength += d[i].length;
        })
        sumLength /= data.length;
        transitionProbs.push([1 / sumLength, 1 - 1 / sumLength])
    }
    return {
        data,
        emissionProbs: curMs,
        transitionProbs
    }
}

function gauss(num, ms) {
    let mean = ms[0];
    let std = ms[1];
    let exp = Math.exp((-0.5) * (num - mean) ** 2 / (std ** 2))
    return (2 * Math.PI * std ** 2) ** (-0.5) * exp;
}

function viterbi(evidence, stateCount, transitionProbs, emissionParams) {
    let probability = 0.0;
    let T = evidence.length;
    let K = stateCount;

    let table = [];
    for (let i = 0; i < stateCount; i++) {
        table.push(new Array(T).fill(0));
    }
    for (let j = 0; j < T; j++) {
        for (let i = 0; i < K; i++) {
            let eProb = gauss(evidence[j], emissionParams[i]);
            if (j == 0) {
                table[i][j] = eProb;
                continue;
            }
            let bestProb = Math.max(table[i][j - 1] * transitionProbs[i][1], i != 0 ? table[i - 1][j - 1] * transitionProbs[i - 1][0] : -1) * eProb;
            table[i][j] = bestProb;
        }
    }

    for (let i = 0; i < K; i++) {
        if (table[i][T - 1] > probability) {
            probability = table[i][T - 1];
        }
    }
    return probability;

}

function trainAll() {
    lettersData = lettersData.map(data => {
        return trainLetter(data);
    });
}

function trainLetter(data) {
    let trainX =  train(JSON.parse(JSON.stringify(data.initialData[0])), data.stateCount[0])
    let trainY =  train(JSON.parse(JSON.stringify(data.initialData[1])), data.stateCount[1])
    Object.assign(data, {
        data: [trainX.data, trainY.data],
        emissionProbs: [trainX.emissionProbs, trainY.emissionProbs],
        transitionProbs: [trainX.transitionProbs, trainY.transitionProbs]
    });
    return data;
}

trainAll();
console.log(lettersData)
