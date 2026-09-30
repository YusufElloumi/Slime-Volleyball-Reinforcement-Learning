const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');

function loadScript(filePath) {
  const absolutePath = path.resolve(ROOT, filePath);
  const source = fs.readFileSync(absolutePath, 'utf8');
  vm.runInThisContext(source, { filename: absolutePath });
}

function makeGameState() {
  globalThis.gameWidth = 1000;
  globalThis.gameHeight = 500;
  globalThis.slime1 = { x: 200, y: 0, radius: 50, velocityX: 0, velocityY: 0 };
  globalThis.slime2 = { x: 800, y: 0, radius: 50, velocityX: 0, velocityY: 0 };
  globalThis.ball = { x: 500, y: 200, radius: 15, velocityX: 0, velocityY: 0 };
  globalThis.slime1Score = 0;
  globalThis.slime2Score = 0;
  globalThis.slimeLeft = globalThis.slime1;
  globalThis.slimeRight = globalThis.slime2;
}

function initRound(server) {
  globalThis.ball.x = server ? 200 : 800;
  globalThis.ball.y = 200;
  globalThis.ball.velocityX = server ? 4 : -4;
  globalThis.ball.velocityY = 1;

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
  globalThis.applyActionToPlayer = applyActionToPlayer;
  globalThis.makeGameState = makeGameState;
  makeGameState();
  initRound(true);
}

loadScript('SlimeAI.js');
attachHeadlessGameGlobals();
loadScript('RLEnvironment.js');

function createWhiteSlimeAI() {
  const ai = newSlimeAI(false, 'Pathetic White Slime');
  setPatheticWhiteSlime(ai);
  return {
    onReset: function() {
      globalThis.slimeLeft = globalThis.slime1;
      globalThis.slimeRight = globalThis.slime2;
    },
    move: function() {
      globalThis.slimeLeft = globalThis.slime1;
      globalThis.slimeRight = globalThis.slime2;
      ai.move();
      if (ai.movement === 1) {
        globalThis.slime2.velocityX = -8;
      } else if (ai.movement === 2) {
        globalThis.slime2.velocityX = 8;
      } else {
        globalThis.slime2.velocityX = 0;
      }
      if (ai.jumpSet && globalThis.slime2.y === 0) {
        globalThis.slime2.velocityY = 18;
      }
    }
  };
}

const env = globalThis.RLEnvironment.createEnvironment({ opponentController: createWhiteSlimeAI() });

function processStep(action) {
  const result = env.step(action);
  return result;
}

function respond(payload) {
  process.stdout.write(JSON.stringify(payload) + '\n');
}

process.stdin.setEncoding('utf8');
let buffer = '';
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let newlineIndex;
  while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
    const raw = buffer.slice(0, newlineIndex).trim();
    buffer = buffer.slice(newlineIndex + 1);
    if (!raw) continue;
    try {
      const message = JSON.parse(raw);
      const cmd = message.cmd || 'noop';
      if (cmd === 'reset') {
        respond({ state: env.reset(Boolean(message.server_side), Boolean(message.reset_scores)) });
      } else if (cmd === 'get_state') {
        respond({ state: env.get_state(message.side || 'left') });
      } else if (cmd === 'step') {
        respond(processStep(message.action));
      } else if (cmd === 'quit') {
        process.exit(0);
      } else {
        respond({ error: 'unknown command' });
      }
    } catch (error) {
      respond({ error: String(error && error.message ? error.message : error) });
    }
  }
});

respond({ ready: true });
