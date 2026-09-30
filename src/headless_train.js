const fs = require('fs');

function makeGameState() {
  globalThis.slime1 = { x: 200, y: 0, radius: 50, velocityX: 0, velocityY: 0 };
  globalThis.slime2 = { x: 800, y: 0, radius: 50, velocityX: 0, velocityY: 0 };
  globalThis.ball = { x: 500, y: 200, radius: 15, velocityX: 0, velocityY: 0 };
  globalThis.slime1Score = 0;
  globalThis.slime2Score = 0;
}

function initRound(server) {
  globalThis.ball.x = server ? 200 : 800;
  globalThis.ball.y = 200;
  globalThis.ball.velocityX = 0;
  globalThis.ball.velocityY = 0;

  globalThis.slime1.x = 200;
  globalThis.slime1.y = 0;
  globalThis.slime1.velocityX = 0;
  globalThis.slime1.velocityY = 0;

  globalThis.slime2.x = 800;
  globalThis.slime2.y = 0;
  globalThis.slime2.velocityX = 0;
  globalThis.slime2.velocityY = 0;
}

function updateSlime(s, leftLimit, rightLimit) {
  if (s.velocityX !== 0) {
    s.x += s.velocityX;
    if (s.x - s.radius < leftLimit) {
      s.x = leftLimit + s.radius;
    } else if (s.x + s.radius > rightLimit) {
      s.x = rightLimit - s.radius;
    }
  }

  if (s.velocityY !== 0 || s.y > 0) {
    s.y += s.velocityY;
    s.velocityY -= 1.2;
    if (s.y < 0) {
      s.y = 0;
      s.velocityY = 0;
    }
  }
}

function collisionBallSlime(s) {
  const dx = globalThis.ball.x - s.x;
  const dy = globalThis.ball.y - s.y;
  const dist = Math.sqrt(dx * dx + dy * dy);

  const dVelocityX = globalThis.ball.velocityX - s.velocityX;
  const dVelocityY = globalThis.ball.velocityY - s.velocityY;

  if (dy > 0 && dist < globalThis.ball.radius + s.radius) {
    globalThis.ball.x = s.x + (s.radius + globalThis.ball.radius) * dx / dist;
    globalThis.ball.y = s.y + (s.radius + globalThis.ball.radius) * dy / dist;

    const something = (dx * dVelocityX + dy * dVelocityY) / dist;
    if (something <= 0) {
      globalThis.ball.velocityX += s.velocityX - 1.92 * dx * something / dist;
      globalThis.ball.velocityY += s.velocityY - 1.92 * dy * something / dist;
      if (globalThis.ball.velocityX < -15) globalThis.ball.velocityX = -15;
      else if (globalThis.ball.velocityX > 15) globalThis.ball.velocityX = 15;
      if (globalThis.ball.velocityY < -11) globalThis.ball.velocityY = -11;
      else if (globalThis.ball.velocityY > 11) globalThis.ball.velocityY = 11;
    }
  }
}

function updateBall() {
  globalThis.ball.x += globalThis.ball.velocityX;
  globalThis.ball.y += globalThis.ball.velocityY;
  globalThis.ball.velocityY += -0.5;

  if (globalThis.ball.velocityY < -11) {
    globalThis.ball.velocityY = -11;
  }

  collisionBallSlime(globalThis.slime1);
  collisionBallSlime(globalThis.slime2);

  if (globalThis.ball.x - globalThis.ball.radius < 0) {
    globalThis.ball.x = globalThis.ball.radius;
    globalThis.ball.velocityX = -globalThis.ball.velocityX;
  } else if (globalThis.ball.x + globalThis.ball.radius > 1000) {
    globalThis.ball.x = 1000 - globalThis.ball.radius;
    globalThis.ball.velocityX = -globalThis.ball.velocityX;
  }

  if (globalThis.ball.y < 0) {
    if (globalThis.ball.x > 500) {
      globalThis.slime1Score += 1;
      initRound(true);
    } else {
      globalThis.slime2Score += 1;
      initRound(false);
    }
  }
}

function applyActionToPlayer(action, target) {
  const player = target || globalThis.slime1;
  if (action === 1) {
    player.velocityX = -8;
  } else if (action === 2) {
    player.velocityX = 8;
  } else if (action === 3) {
    player.velocityX = 0;
    if (player.y === 0) {
      player.velocityY = 18;
    }
  } else {
    player.velocityX = 0;
  }
}

function attachHeadlessGameGlobals() {
  globalThis.initRound = initRound;
  globalThis.updateSlime = updateSlime;
  globalThis.updateBall = updateBall;
  globalThis.initRound(true);
}

function defaultOptions() {
  return {
    alpha: 0.2,
    gamma: 0.95,
    epsilon: 1.0,
    minEpsilon: 0.15,
    epsilonDecay: 0.999,
    optimisticValue: 0.5,
    episodes: 1000,
    logEvery: 50,
    savePath: './agents_headless.json'
  };
}

function parseArgs(argv) {
  const out = defaultOptions();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--episodes') out.episodes = Number(argv[i + 1] || out.episodes);
    if (arg === '--log-every') out.logEvery = Number(argv[i + 1] || out.logEvery);
    if (arg === '--epsilon') out.epsilon = Number(argv[i + 1] || out.epsilon);
    if (arg === '--min-epsilon') out.minEpsilon = Number(argv[i + 1] || out.minEpsilon);
    if (arg === '--save') out.savePath = argv[i + 1] || out.savePath;
    if (arg === '--resume') out.resume = true;
  }
  return out;
}

function saveAgents(leftAgent, rightAgent, savePath) {
  const snapshot = {
    left: {
      qTable: leftAgent.qTable,
      epsilon: leftAgent.getEpsilon()
    },
    right: {
      qTable: rightAgent.qTable,
      epsilon: rightAgent.getEpsilon()
    },
    savedAt: new Date().toISOString()
  };
  fs.writeFileSync(savePath, JSON.stringify(snapshot, null, 2));
  console.log('Saved agent state to', savePath);
}

function loadAgents(savePath, env, leftOptions, rightOptions) {
  if (!fs.existsSync(savePath)) {
    return {
      leftAgent: globalThis.RLEnvironment.createOptimisticEpsilonGreedyAgent(env, leftOptions),
      rightAgent: globalThis.RLEnvironment.createOptimisticEpsilonGreedyAgent(env, rightOptions)
    };
  }

  const raw = JSON.parse(fs.readFileSync(savePath, 'utf8'));
  const leftAgent = globalThis.RLEnvironment.createOptimisticEpsilonGreedyAgent(env, leftOptions);
  const rightAgent = globalThis.RLEnvironment.createOptimisticEpsilonGreedyAgent(env, rightOptions);

  leftAgent.qTable = raw.left && raw.left.qTable ? raw.left.qTable : {};
  rightAgent.qTable = raw.right && raw.right.qTable ? raw.right.qTable : {};
  leftAgent.setEpsilon(raw.left && raw.left.epsilon !== undefined ? raw.left.epsilon : leftOptions.epsilon);
  rightAgent.setEpsilon(raw.right && raw.right.epsilon !== undefined ? raw.right.epsilon : rightOptions.epsilon);

  console.log('Loaded agent state from', savePath);
  return { leftAgent, rightAgent };
}

function trainOvernight(options) {
  makeGameState();
  attachHeadlessGameGlobals();

  require('./RLEnvironment.js');

  const env = globalThis.RLEnvironment.createEnvironment();
  const leftOptions = {
    alpha: options.alpha,
    gamma: options.gamma,
    epsilon: options.epsilon,
    minEpsilon: options.minEpsilon,
    epsilonDecay: options.epsilonDecay,
    optimisticValue: options.optimisticValue
  };
  const rightOptions = {
    alpha: options.alpha,
    gamma: options.gamma,
    epsilon: options.epsilon,
    minEpsilon: options.minEpsilon,
    epsilonDecay: options.epsilonDecay,
    optimisticValue: options.optimisticValue
  };

  let started = loadAgents(options.savePath, env, leftOptions, rightOptions);
  const leftAgent = started.leftAgent;
  const rightAgent = started.rightAgent;

  let leftWins = 0;
  let rightWins = 0;

  for (let ep = 1; ep <= options.episodes; ep++) {
    const serverSide = (ep % 2 === 0);
    globalThis.slime1Score = 0;
    globalThis.slime2Score = 0;
    let state = env.reset(serverSide, true);
    let steps = 0;
    let leftRewardTotal = 0;
    let rightRewardTotal = 0;

    while (true) {
      const leftState = env.get_state_for_side('left');
      const rightState = env.get_state_for_side('right');
      const leftAction = leftAgent.chooseAction(leftState);
      const rightAction = rightAgent.chooseAction(rightState);
      const result = env.step([leftAction, rightAction]);

      leftAgent.update(leftState, leftAction, result.rewardLeft || 0, env.get_state_for_side('left'), false);
      rightAgent.update(rightState, rightAction, result.rewardRight || 0, env.get_state_for_side('right'), false);

      leftRewardTotal += result.rewardLeft || 0;
      rightRewardTotal += result.rewardRight || 0;
      steps += 1;
      state = result.next_state;

      if (result.done) {
        break;
      }
    }

    if (globalThis.slime1Score > globalThis.slime2Score) leftWins += 1;
    else if (globalThis.slime2Score > globalThis.slime1Score) rightWins += 1;

    if (ep % options.logEvery === 0 || ep === 1) {
      console.log('EP', ep, 'leftWins', leftWins, 'rightWins', rightWins, 'steps', steps,
        'leftReward', leftRewardTotal, 'rightReward', rightRewardTotal,
        'leftEps', leftAgent.getEpsilon(), 'rightEps', rightAgent.getEpsilon());
    }

    if (ep % 200 === 0) {
      saveAgents(leftAgent, rightAgent, options.savePath);
    }
  }

  saveAgents(leftAgent, rightAgent, options.savePath);
  console.log('Training finished. Final leftWins=', leftWins, 'rightWins=', rightWins);
  return { leftAgent, rightAgent };
}

const args = parseArgs(process.argv.slice(2));
trainOvernight(args);
