/**
 * NEON SNAKE - Arcade Edition
 * A retro-futuristic snake game with CRT effects
 */

// Game Configuration
const CONFIG = {
    GRID_SIZE: 20,
    CELL_SIZE: 20,
    CANVAS_WIDTH: 600,
    CANVAS_HEIGHT: 400,
    INITIAL_SPEED: 150,
    MIN_SPEED: 60,
    SPEED_DECREMENT: 5,
    COLORS: {
        SNAKE_HEAD: '#00ff41',
        SNAKE_BODY: '#00cc33',
        SNAKE_GLOW: '#00ff41',
        FOOD: '#ff0080',
        FOOD_GLOW: '#ff0080',
        GRID: 'rgba(0, 255, 65, 0.05)',
        BACKGROUND: '#050505'
    }
};

// Game State
const state = {
    snake: [],
    direction: { x: 1, y: 0 },
    nextDirection: { x: 1, y: 0 },
    food: null,
    score: 0,
    highScore: parseInt(localStorage.getItem('snakeHighScore')) || 0,
    level: 1,
    speed: CONFIG.INITIAL_SPEED,
    gameLoop: null,
    isGameOver: false,
    isPaused: false,
    isStarted: false,
    particles: []
};

// DOM Elements
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreEl = document.getElementById('score');
const highScoreEl = document.getElementById('highScore');
const levelEl = document.getElementById('level');
const finalScoreEl = document.getElementById('finalScore');
const gameOverScreen = document.getElementById('gameOverScreen');
const startScreen = document.getElementById('startScreen');
const restartBtn = document.getElementById('restartBtn');
const startBtn = document.getElementById('startBtn');

// Initialize
function init() {
    // Set canvas size
    canvas.width = CONFIG.CANVAS_WIDTH;
    canvas.height = CONFIG.CANVAS_HEIGHT;
    
    // Update high score display
    highScoreEl.textContent = state.highScore.toString().padStart(4, '0');
    
    // Event listeners
    document.addEventListener('keydown', handleKeyDown);
    restartBtn.addEventListener('click', restartGame);
    startBtn.addEventListener('click', startGame);
    
    // Show start screen
    startScreen.classList.add('active');
    
    // Initial render
    draw();
}

// Start Game
function startGame() {
    startScreen.classList.remove('active');
    state.isStarted = true;
    resetGame();
    startGameLoop();
}

// Reset Game State
function resetGame() {
    // Initialize snake in the center
    const startX = Math.floor(CONFIG.CANVAS_WIDTH / CONFIG.CELL_SIZE / 2);
    const startY = Math.floor(CONFIG.CANVAS_HEIGHT / CONFIG.CELL_SIZE / 2);
    
    state.snake = [
        { x: startX, y: startY },
        { x: startX - 1, y: startY },
        { x: startX - 2, y: startY }
    ];
    
    state.direction = { x: 1, y: 0 };
    state.nextDirection = { x: 1, y: 0 };
    state.score = 0;
    state.level = 1;
    state.speed = CONFIG.INITIAL_SPEED;
    state.isGameOver = false;
    state.isPaused = false;
    state.particles = [];
    
    spawnFood();
    updateUI();
}

// Spawn Food
function spawnFood() {
    const maxX = CONFIG.CANVAS_WIDTH / CONFIG.CELL_SIZE;
    const maxY = CONFIG.CANVAS_HEIGHT / CONFIG.CELL_SIZE;
    
    do {
        state.food = {
            x: Math.floor(Math.random() * maxX),
            y: Math.floor(Math.random() * maxY)
        };
    } while (isSnakeAt(state.food.x, state.food.y));
}

// Check if snake is at position
function isSnakeAt(x, y) {
    return state.snake.some(segment => segment.x === x && segment.y === y);
}

// Handle Keyboard Input
function handleKeyDown(e) {
    if (!state.isStarted) {
        if (e.code === 'Space') {
            startGame();
        }
        return;
    }
    
    if (state.isGameOver) {
        if (e.code === 'Space') {
            restartGame();
        }
        return;
    }
    
    switch(e.key) {
        case 'ArrowUp':
            if (state.direction.y === 0) {
                state.nextDirection = { x: 0, y: -1 };
            }
            e.preventDefault();
            break;
        case 'ArrowDown':
            if (state.direction.y === 0) {
                state.nextDirection = { x: 0, y: 1 };
            }
            e.preventDefault();
            break;
        case 'ArrowLeft':
            if (state.direction.x === 0) {
                state.nextDirection = { x: -1, y: 0 };
            }
            e.preventDefault();
            break;
        case 'ArrowRight':
            if (state.direction.x === 0) {
                state.nextDirection = { x: 1, y: 0 };
            }
            e.preventDefault();
            break;
        case ' ':
            togglePause();
            e.preventDefault();
            break;
    }
}

// Toggle Pause
function togglePause() {
    if (!state.isStarted || state.isGameOver) return;
    state.isPaused = !state.isPaused;
}

// Game Loop
function startGameLoop() {
    if (state.gameLoop) clearInterval(state.gameLoop);
    state.gameLoop = setInterval(gameStep, state.speed);
}

// Game Step
function gameStep() {
    if (state.isPaused || state.isGameOver) return;
    
    // Update direction
    state.direction = { ...state.nextDirection };
    
    // Calculate new head position
    const head = { ...state.snake[0] };
    head.x += state.direction.x;
    head.y += state.direction.y;
    
    // Check wall collision
    const maxX = CONFIG.CANVAS_WIDTH / CONFIG.CELL_SIZE;
    const maxY = CONFIG.CANVAS_HEIGHT / CONFIG.CELL_SIZE;
    
    if (head.x < 0 || head.x >= maxX || head.y < 0 || head.y >= maxY) {
        gameOver();
        return;
    }
    
    // Check self collision
    if (isSnakeAt(head.x, head.y)) {
        gameOver();
        return;
    }
    
    // Move snake
    state.snake.unshift(head);
    
    // Check food collision
    if (head.x === state.food.x && head.y === state.food.y) {
        // Eat food
        state.score += 10 * state.level;
        createParticles(
            state.food.x * CONFIG.CELL_SIZE + CONFIG.CELL_SIZE / 2,
            state.food.y * CONFIG.CELL_SIZE + CONFIG.CELL_SIZE / 2,
            CONFIG.COLORS.FOOD
        );
        
        // Level up every 50 points
        if (state.score > 0 && state.score % 50 === 0) {
            state.level++;
            state.speed = Math.max(CONFIG.MIN_SPEED, state.speed - CONFIG.SPEED_DECREMENT);
            startGameLoop();
        }
        
        spawnFood();
        updateUI();
    } else {
        // Remove tail
        state.snake.pop();
    }
    
    // Update particles
    updateParticles();
    
    // Render
    draw();
}

// Create Particle Effect
function createParticles(x, y, color) {
    for (let i = 0; i < 12; i++) {
        const angle = (Math.PI * 2 * i) / 12;
        const speed = 2 + Math.random() * 3;
        state.particles.push({
            x: x,
            y: y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 1,
            color: color
        });
    }
}

// Update Particles
function updateParticles() {
    state.particles = state.particles.filter(p => {
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 0.05;
        return p.life > 0;
    });
}

// Game Over
function gameOver() {
    state.isGameOver = true;
    clearInterval(state.gameLoop);
    
    // Update high score
    if (state.score > state.highScore) {
        state.highScore = state.score;
        localStorage.setItem('snakeHighScore', state.highScore);
        highScoreEl.textContent = state.highScore.toString().padStart(4, '0');
    }
    
    finalScoreEl.textContent = state.score;
    gameOverScreen.classList.add('active');
}

// Restart Game
function restartGame() {
    gameOverScreen.classList.remove('active');
    resetGame();
    startGameLoop();
}

// Update UI
function updateUI() {
    scoreEl.textContent = state.score.toString().padStart(4, '0');
    levelEl.textContent = state.level.toString().padStart(2, '0');
}

// Draw Game
function draw() {
    // Clear canvas
    ctx.fillStyle = CONFIG.COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    drawGrid();
    
    // Draw food
    drawFood();
    
    // Draw snake
    drawSnake();
    
    // Draw particles
    drawParticles();
    
    // Draw pause overlay
    if (state.isPaused) {
        drawPauseOverlay();
    }
}

// Draw Grid
function drawGrid() {
    ctx.strokeStyle = CONFIG.COLORS.GRID;
    ctx.lineWidth = 1;
    
    for (let x = 0; x <= CONFIG.CANVAS_WIDTH; x += CONFIG.CELL_SIZE) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, CONFIG.CANVAS_HEIGHT);
        ctx.stroke();
    }
    
    for (let y = 0; y <= CONFIG.CANVAS_HEIGHT; y += CONFIG.CELL_SIZE) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(CONFIG.CANVAS_WIDTH, y);
        ctx.stroke();
    }
}

// Draw Snake
function drawSnake() {
    state.snake.forEach((segment, index) => {
        const x = segment.x * CONFIG.CELL_SIZE;
        const y = segment.y * CONFIG.CELL_SIZE;
        const size = CONFIG.CELL_SIZE - 2;
        
        // Glow effect
        ctx.shadowBlur = 15;
        ctx.shadowColor = CONFIG.COLORS.SNAKE_GLOW;
        
        if (index === 0) {
            // Head
            ctx.fillStyle = CONFIG.COLORS.SNAKE_HEAD;
            ctx.fillRect(x + 1, y + 1, size, size);
            
            // Eyes
            ctx.shadowBlur = 0;
            ctx.fillStyle = '#000';
            const eyeSize = 3;
            const eyeOffset = 4;
            
            if (state.direction.x === 1) {
                ctx.fillRect(x + size - 2, y + eyeOffset, eyeSize, eyeSize);
                ctx.fillRect(x + size - 2, y + size - eyeOffset - 3, eyeSize, eyeSize);
            } else if (state.direction.x === -1) {
                ctx.fillRect(x + 3, y + eyeOffset, eyeSize, eyeSize);
                ctx.fillRect(x + 3, y + size - eyeOffset - 3, eyeSize, eyeSize);
            } else if (state.direction.y === -1) {
                ctx.fillRect(x + eyeOffset, y + 3, eyeSize, eyeSize);
                ctx.fillRect(x + size - eyeOffset - 3, y + 3, eyeSize, eyeSize);
            } else {
                ctx.fillRect(x + eyeOffset, y + size - 5, eyeSize, eyeSize);
                ctx.fillRect(x + size - eyeOffset - 3, y + size - 5, eyeSize, eyeSize);
            }
        } else {
            // Body
            const alpha = 1 - (index / state.snake.length) * 0.5;
            ctx.fillStyle = `rgba(0, 204, 51, ${alpha})`;
            ctx.fillRect(x + 1, y + 1, size, size);
        }
        
        ctx.shadowBlur = 0;
    });
}

// Draw Food
function drawFood() {
    if (!state.food) return;
    
    const x = state.food.x * CONFIG.CELL_SIZE;
    const y = state.food.y * CONFIG.CELL_SIZE;
    const centerX = x + CONFIG.CELL_SIZE / 2;
    const centerY = y + CONFIG.CELL_SIZE / 2;
    
    // Pulsing effect
    const pulse = 1 + Math.sin(Date.now() / 200) * 0.1;
    
    // Glow
    ctx.shadowBlur = 20 * pulse;
    ctx.shadowColor = CONFIG.COLORS.FOOD_GLOW;
    
    // Draw food as a diamond shape
    ctx.fillStyle = CONFIG.COLORS.FOOD;
    ctx.beginPath();
    const size = (CONFIG.CELL_SIZE - 4) / 2 * pulse;
    ctx.moveTo(centerX, centerY - size);
    ctx.lineTo(centerX + size, centerY);
    ctx.lineTo(centerX, centerY + size);
    ctx.lineTo(centerX - size, centerY);
    ctx.closePath();
    ctx.fill();
    
    ctx.shadowBlur = 0;
}

// Draw Particles
function drawParticles() {
    state.particles.forEach(p => {
        ctx.globalAlpha = p.life;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fill();
    });
    ctx.globalAlpha = 1;
}

// Draw Pause Overlay
function drawPauseOverlay() {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.font = '30px "Press Start 2P"';
    ctx.fillStyle = CONFIG.COLORS.SNAKE_HEAD;
    ctx.textAlign = 'center';
    ctx.shadowBlur = 10;
    ctx.shadowColor = CONFIG.COLORS.SNAKE_GLOW;
    ctx.fillText('PAUSED', canvas.width / 2, canvas.height / 2);
    ctx.shadowBlur = 0;
    
    ctx.font = '16px "VT323"';
    ctx.fillStyle = CONFIG.COLORS.FOOD;
    ctx.fillText('PRESS SPACE TO RESUME', canvas.width / 2, canvas.height / 2 + 40);
}

// Start the game
init();