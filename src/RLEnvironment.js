(function(global) {
  var ACTION_NONE = 0;
  var ACTION_LEFT = 1;
  var ACTION_RIGHT = 2;
  var ACTION_JUMP = 3;

  function ensureGameState() {
    if (typeof slime1 === 'undefined' || typeof slime2 === 'undefined') {
      throw new Error('RLEnvironment requires slime1 and slime2 to exist on the page.');
    }
    if (typeof ball === 'undefined') {
      throw new Error('RLEnvironment requires ball to exist on the page.');
    }
    if (typeof updateBall !== 'function' || typeof updateSlime !== 'function' || typeof initRound !== 'function') {
      throw new Error('RLEnvironment requires the existing game update functions to be available.');
    }
  }

  function applyActionToPlayer(action, target) {
    var player = target || slime1;
    if (action === ACTION_LEFT) {
      player.velocityX = -8;
    } else if (action === ACTION_RIGHT) {
      player.velocityX = 8;
    } else if (action === ACTION_JUMP) {
      player.velocityX = 0;
      if (player.y === 0) {
        player.velocityY = 18;
      }
    } else {
      player.velocityX = 0;
    }
  }

  function applyOpponentController(controller) {
    if (!controller || typeof controller.move !== 'function') {
      return;
    }
    slimeLeft = slime1;
    slimeRight = slime2;
    controller.move();

    if (controller.movement === 1) {
      slime2.velocityX = -8;
    } else if (controller.movement === 2) {
      slime2.velocityX = 8;
    } else {
      slime2.velocityX = 0;
    }

    if (controller.jumpSet && slime2.y === 0) {
      slime2.velocityY = 18;
    }
  }

  function getStateForSide(side) {
    ensureGameState();

    var playerSide = side === 'right' ? 'right' : 'left';
    var player = playerSide === 'right' ? slime2 : slime1;
    var opponent = playerSide === 'right' ? slime1 : slime2;
    var relativeBallX = ball.x - player.x;
    var relativeBallY = ball.y - player.y;
    var relativeOpponentX = opponent.x - player.x;
    var relativeOpponentY = opponent.y - player.y;
    var relativeBallVX = ball.velocityX;
    var relativeBallVY = ball.velocityY;

    if (playerSide === 'right') {
      relativeBallX *= -1;
      relativeOpponentX *= -1;
      relativeBallVX *= -1;
    }

    var scorePlayer = playerSide === 'right' ? slime2Score : slime1Score;
    var scoreOpponent = playerSide === 'right' ? slime1Score : slime2Score;

    return {
      side: playerSide,
      player_x: player.x,
      player_y: player.y,
      player_velocity_x: player.velocityX,
      player_velocity_y: player.velocityY,
      ball_x: relativeBallX,
      ball_y: relativeBallY,
      ball_velocity_x: relativeBallVX,
      ball_velocity_y: relativeBallVY,
      relative_ball_x: relativeBallX,
      relative_ball_y: relativeBallY,
      relative_ball_vx: relativeBallVX,
      relative_ball_vy: relativeBallVY,
      ball_distance: Math.hypot(relativeBallX, relativeBallY),
      opponent_x: relativeOpponentX,
      opponent_y: relativeOpponentY,
      opponent_velocity_x: playerSide === 'right' ? -opponent.velocityX : opponent.velocityX,
      opponent_velocity_y: opponent.velocityY,
      player_on_ground: player.y === 0,
      score_player: scorePlayer,
      score_opponent: scoreOpponent,
      score_diff: scorePlayer - scoreOpponent,
      game_state: 'running'
    };
  }

  function getState() {
    return getStateForSide('left');
  }

  function defaultActionSpace() {
    return [ACTION_NONE, ACTION_LEFT, ACTION_RIGHT];
  }

  function discretize(value, bucketSize) {
    if (typeof value !== 'number' || isNaN(value)) {
      return 0;
    }
    if (bucketSize <= 0) {
      return 0;
    }
    return Math.floor(value / bucketSize);
  }

  function stateKeyFromState(state) {
    var ballX = state.relative_ball_x !== undefined ? state.relative_ball_x : state.ball_x;
    var ballY = state.relative_ball_y !== undefined ? state.relative_ball_y : state.ball_y;
    var ballVX = state.relative_ball_vx !== undefined ? state.relative_ball_vx : state.ball_velocity_x;
    var ballVY = state.relative_ball_vy !== undefined ? state.relative_ball_vy : state.ball_velocity_y;
    var opponentX = state.opponent_x !== undefined ? state.opponent_x : state.relative_opponent_x;
    var playerOnGround = state.player_on_ground !== undefined ? state.player_on_ground : (state.player_y === 0);
    var scoreDiff = state.score_diff !== undefined ? state.score_diff : (state.score_player - state.score_opponent);

    return [
      discretize(state.player_x, 50),
      discretize(ballX, 25),
      discretize(ballY, 20),
      discretize(ballVX, 2),
      discretize(ballVY, 2),
      discretize(opponentX, 50),
      discretize(state.player_y, 20),
      playerOnGround ? 1 : 0,
      discretize(scoreDiff, 1),
      discretize(state.ball_distance !== undefined ? state.ball_distance : Math.hypot(ballX, ballY), 25)
    ].join('|');
  }

  function startLiveTrainingVsAI(options) {
    options = options || {};
    var ai = options.ai || null;
    var agentOptions = options.agent || {};
    var stepMs = options.stepMs || 100;
    var env = createEnvironment({ opponentController: ai });
    var agent = createOptimisticEpsilonGreedyAgent(env, agentOptions);
    var state = env.reset(true);
    var episode = 0;
    var totalReward = 0;
    var steps = 0;

    var intervalId = setInterval(function() {
      var action = agent.chooseAction(state);
      var result = env.step(action);
      totalReward += result.reward;
      steps += 1;
      agent.update(state, action, result.reward, result.next_state, result.done);
      state = result.next_state;

      if (result.done) {
        episode += 1;
        console.log('EP ' + episode + ' reward=' + totalReward + ' steps=' + steps + ' epsilon=' + agent.getEpsilon());
        totalReward = 0;
        steps = 0;
        state = env.reset(true);
      }
    }, stepMs);

    return {
      env: env,
      agent: agent,
      stop: function() {
        clearInterval(intervalId);
      }
    };
  }

  function startTwoAgentTraining(options) {
    options = options || {};
    var leftAgentOptions = options.leftAgent || {};
    var rightAgentOptions = options.rightAgent || {};
    var stepMs = options.stepMs || 100;
    var env = createEnvironment();
    var leftAgent = createOptimisticEpsilonGreedyAgent(env, leftAgentOptions);
    var rightAgent = createOptimisticEpsilonGreedyAgent(env, rightAgentOptions);
    var state = env.reset(true);
    var episode = 0;
    var leftWins = 0;
    var rightWins = 0;
    var steps = 0;

    var intervalId = setInterval(function() {
      var leftState = env.get_state_for_side('left');
      var rightState = env.get_state_for_side('right');
      var leftAction = leftAgent.chooseAction(leftState);
      var rightAction = rightAgent.chooseAction(rightState);
      var result = env.step([leftAction, rightAction]);

      leftAgent.update(leftState, leftAction, result.rewardLeft || 0, env.get_state_for_side('left'), false);
      rightAgent.update(rightState, rightAction, result.rewardRight || 0, env.get_state_for_side('right'), false);

      steps += 1;
      state = env.get_state_for_side('left');

      if (result.done) {
        episode += 1;
        if (slime1Score > slime2Score) {
          leftWins += 1;
        } else if (slime2Score > slime1Score) {
          rightWins += 1;
        }
        console.log('EP ' + episode + ' leftWins=' + leftWins + ' rightWins=' + rightWins + ' steps=' + steps +
          ' leftEps=' + leftAgent.getEpsilon() + ' rightEps=' + rightAgent.getEpsilon());
        steps = 0;
        state = env.reset(true);
      }
    }, stepMs);

    return {
      env: env,
      leftAgent: leftAgent,
      rightAgent: rightAgent,
      stop: function() {
        clearInterval(intervalId);
      }
    };
  }

  function createEnvironment(options) {
    options = options || {};
    var opponentController = options.opponentController || null;

    return {
      action_space: defaultActionSpace(),
      reset: function(serverSide, resetScores) {
        ensureGameState();
        var server = (typeof serverSide === 'boolean') ? serverSide : true;
        var shouldResetScores = (typeof resetScores === 'boolean') ? resetScores : true;

        if (shouldResetScores) {
          slime1Score = 0;
          slime2Score = 0;
        }

        initRound(server);
        ball.velocityX = server ? 4 : -4;
        ball.velocityY = 1;
        if (opponentController && typeof opponentController.onReset === 'function') {
          opponentController.onReset();
        }
        return getState();
      },
      get_state: function(side) {
        return getStateForSide(side || 'left');
      },
      get_state_for_side: function(side) {
        return getStateForSide(side || 'left');
      },
      step: function(action) {
        ensureGameState();

        var leftAction = action;
        var rightAction = ACTION_NONE;

        if (Array.isArray(action)) {
          leftAction = action[0];
          rightAction = action[1];
        }

        if (leftAction !== ACTION_NONE && leftAction !== ACTION_LEFT && leftAction !== ACTION_RIGHT && leftAction !== ACTION_JUMP) {
          throw new Error('Left action must be 0 (noop), 1 (left), 2 (right), or 3 (jump) for compatibility, but RL uses only 0/1/2.');
        }
        if (rightAction !== ACTION_NONE && rightAction !== ACTION_LEFT && rightAction !== ACTION_RIGHT && rightAction !== ACTION_JUMP) {
          throw new Error('Right action must be 0 (noop), 1 (left), 2 (right), or 3 (jump) for compatibility, but RL uses only 0/1/2.');
        }

        var previousPlayerScore = slime1Score;
        var previousOpponentScore = slime2Score;
        var prevBallX = ball.x;
        var prevBallY = ball.y;

        applyActionToPlayer(leftAction, slime1);
        applyActionToPlayer(rightAction, slime2);
        if (opponentController) {
          applyOpponentController(opponentController);
        }

        updateSlime(slime1, 0, 495);
        updateSlime(slime2, 505, 1000);
        updateBall();

        var clearDetected = false;
        var clearSide = null;
        if (prevBallX < 500 && ball.x >= 500 && ball.velocityX >= 0 && ball.y > 0) {
          clearDetected = true;
          clearSide = 'left';
        } else if (prevBallX > 500 && ball.x <= 500 && ball.velocityX <= 0 && ball.y > 0) {
          clearDetected = true;
          clearSide = 'right';
        }

        var reward = 0;
        var rewardLeft = 0;
        var rewardRight = 0;

        if (clearDetected) {
          if (clearSide === 'left') {
            rewardLeft += 1.0;
            rewardRight -= 0.0;
          } else {
            rewardRight += 1.0;
            rewardLeft -= 0.0;
          }
        }

        if (slime1Score > previousPlayerScore) {
          rewardLeft += 10.0;
          rewardRight -= 10.0;
        } else if (slime2Score > previousOpponentScore) {
          rewardLeft -= 10.0;
          rewardRight += 10.0;
        }

        reward = rewardLeft;
        var done = (slime1Score !== previousPlayerScore) || (slime2Score !== previousOpponentScore);
        var winner = null;
        if (slime1Score > previousPlayerScore) {
          winner = 'left';
        } else if (slime2Score > previousOpponentScore) {
          winner = 'right';
        }

        return {
          next_state: getState(),
          reward: reward,
          rewardLeft: rewardLeft,
          rewardRight: rewardRight,
          reward_clear: clearDetected ? 1.0 : 0.0,
          clear_detected: clearDetected,
          clear_side: clearSide,
          done: done,
          winner: winner,
          previous_ball_x: prevBallX,
          previous_ball_y: prevBallY,
          ball_x: ball.x,
          ball_y: ball.y
        };
      }
    };
  }

  function createOptimisticEpsilonGreedyAgent(env, options) {
    options = options || {};

    var actionSpace = env && env.action_space ? env.action_space : defaultActionSpace();
    var alpha = options.alpha !== undefined ? options.alpha : 0.25;
    var gamma = options.gamma !== undefined ? options.gamma : 0.95;
    var epsilon = options.epsilon !== undefined ? options.epsilon : 0.25;
    var minEpsilon = options.minEpsilon !== undefined ? options.minEpsilon : 0.05;
    var epsilonDecay = options.epsilonDecay !== undefined ? options.epsilonDecay : 0.995;
    var optimisticValue = options.optimisticValue !== undefined ? options.optimisticValue : 1.5;

    var qTable = {};

    function getActionRow(stateKey) {
      if (!qTable[stateKey]) {
        qTable[stateKey] = {};
        for (var i = 0; i < actionSpace.length; i++) {
          qTable[stateKey][actionSpace[i]] = optimisticValue;
        }
      }
      return qTable[stateKey];
    }

    function getQValue(stateKey, action) {
      var row = qTable[stateKey];
      if (!row) {
        return optimisticValue;
      }
      if (row[action] === undefined) {
        return optimisticValue;
      }
      return row[action];
    }

    function getMaxQValue(stateKey) {
      var row = qTable[stateKey];
      if (!row) {
        return optimisticValue;
      }
      var best = -Infinity;
      for (var i = 0; i < actionSpace.length; i++) {
        var action = actionSpace[i];
        if (row[action] > best) {
          best = row[action];
        }
      }
      return best;
    }

    function chooseAction(state) {
      var stateKey = stateKeyFromState(state);
      if (Math.random() < epsilon) {
        var randomIndex = Math.floor(Math.random() * actionSpace.length);
        return actionSpace[randomIndex];
      }

      var row = getActionRow(stateKey);
      var bestAction = actionSpace[0];
      var bestValue = row[bestAction];
      for (var i = 1; i < actionSpace.length; i++) {
        var action = actionSpace[i];
        if (row[action] > bestValue) {
          bestAction = action;
          bestValue = row[action];
        }
      }
      return bestAction;
    }

    function update(state, action, reward, nextState, done) {
      var stateKey = stateKeyFromState(state);
      var nextStateKey = stateKeyFromState(nextState);
      var row = getActionRow(stateKey);
      var current = row[action];
      var target = reward;

      if (!done) {
        target += gamma * getMaxQValue(nextStateKey);
      }

      row[action] = current + alpha * (target - current);

      if (epsilon > minEpsilon) {
        epsilon = Math.max(minEpsilon, epsilon * epsilonDecay);
      }
    }

    function trainEpisode() {
      var state = env.reset(true);
      var totalReward = 0;
      var steps = 0;

      while (true) {
        var action = chooseAction(state);
        var result = env.step(action);
        totalReward += result.reward;
        update(state, action, result.reward, result.next_state, result.done);
        state = result.next_state;
        steps += 1;

        if (result.done) {
          break;
        }
      }

      return {
        totalReward: totalReward,
        steps: steps,
        epsilon: epsilon
      };
    }

    return {
      qTable: qTable,
      chooseAction: chooseAction,
      update: update,
      trainEpisode: trainEpisode,
      getEpsilon: function() {
        return epsilon;
      },
      setEpsilon: function(value) {
        epsilon = value;
      }
    };
  }

  global.RLEnvironment = {
    ACTION_NONE: ACTION_NONE,
    ACTION_LEFT: ACTION_LEFT,
    ACTION_RIGHT: ACTION_RIGHT,
    ACTION_JUMP: ACTION_JUMP,
    createEnvironment: createEnvironment,
    createOptimisticEpsilonGreedyAgent: createOptimisticEpsilonGreedyAgent,
    startLiveTrainingVsAI: startLiveTrainingVsAI,
    startTwoAgentTraining: startTwoAgentTraining,
    getStateForSide: getStateForSide,
    stateKeyFromState: stateKeyFromState
  };
})(typeof window !== 'undefined' ? window : globalThis);
