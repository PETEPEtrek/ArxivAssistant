#!/bin/bash

# Скрипт для управления ArXiv Assistant в Docker

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка наличия Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker не установлен. Установите Docker и попробуйте снова."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose не установлен. Установите Docker Compose и попробуйте снова."
        exit 1
    fi
    
    print_success "Docker и Docker Compose доступны"
}

# Создание необходимых директорий
create_directories() {
    print_message "Создание необходимых директорий..."
    
    mkdir -p uploaded_pdfs
    mkdir -p paper_rag/data/embeddings
    mkdir -p paper_rag/data/papers
    mkdir -p logs
    mkdir -p models
    
    print_success "Директории созданы"
}

# Запуск основной конфигурации
start() {
    print_message "Запуск ArXiv Assistant..."
    
    docker-compose up -d
    
    print_success "ArXiv Assistant запущен"
    print_message "ArXiv Assistant доступен по адресу: http://localhost:8501"
    print_message "Ollama доступен по адресу: http://localhost:11434"
}

# Запуск с Redis
start_with_redis() {
    print_message "Запуск ArXiv Assistant с Redis..."
    
    docker-compose up -d redis
    docker-compose up -d
    
    print_success "ArXiv Assistant с Redis запущен"
    print_message "ArXiv Assistant доступен по адресу: http://localhost:8501"
    print_message "Ollama доступен по адресу: http://localhost:11434"
    print_message "Redis доступен по адресу: localhost:6379"
}

# Остановка всех сервисов
stop() {
    print_message "Остановка всех сервисов..."
    
    docker-compose down
    
    print_success "Все сервисы остановлены"
}

# Перезапуск сервисов
restart() {
    print_message "Перезапуск сервисов..."
    
    docker-compose restart
    
    print_success "Сервисы перезапущены"
}

# Просмотр логов
logs() {
    if [ -z "$1" ]; then
        print_message "Просмотр логов всех сервисов..."
        docker-compose logs -f
    else
        print_message "Просмотр логов сервиса: $1"
        docker-compose logs -f "$1"
    fi
}

# Очистка
cleanup() {
    print_message "Очистка Docker ресурсов..."
    
    docker-compose down -v
    docker system prune -f
    
    print_success "Очистка завершена"
}

# Сборка образов
build() {
    print_message "Сборка Docker образов..."
    
    docker-compose build --no-cache
    
    print_success "Образы собраны"
}

# Статус сервисов
status() {
    print_message "Статус сервисов..."
    
    docker-compose ps
    
    print_message "Использование ресурсов:"
    docker stats --no-stream
}

# Загрузка модели в Ollama
pull_model() {
    if [ -z "$1" ]; then
        print_error "Укажите название модели (например: qwen3:latest)"
        exit 1
    fi
    
    print_message "Загрузка модели $1 в Ollama..."
    
    docker-compose exec ollama ollama pull "$1"
    
    print_success "Модель $1 загружена"
}

# Показать справку
show_help() {
    echo "ArXiv Assistant Docker Manager"
    echo ""
    echo "Использование: $0 [команда]"
    echo ""
    echo "Команды:"
    echo "  start              - Запуск основной конфигурации"
    echo "  start-redis        - Запуск с Redis"
    echo "  stop               - Остановка всех сервисов"
    echo "  restart            - Перезапуск сервисов"
    echo "  logs [сервис]      - Просмотр логов"
    echo "  build              - Сборка образов"
    echo "  status             - Статус сервисов"
    echo "  cleanup            - Очистка Docker ресурсов"
    echo "  pull-model МОДЕЛЬ  - Загрузка модели в Ollama"
    echo "  help               - Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  $0 start"
    echo "  $0 pull-model qwen3:latest"
    echo "  $0 logs arxiv-assistant"
}

# Основная логика
main() {
    case "${1:-help}" in
        start)
            check_docker
            create_directories
            start
            ;;
        start-redis)
            check_docker
            create_directories
            start_with_redis
            ;;
        stop)
            stop
            ;;
        restart)
            restart
            ;;
        logs)
            logs "$2"
            ;;
        build)
            check_docker
            build
            ;;
        status)
            status
            ;;
        cleanup)
            cleanup
            ;;
        pull-model)
            pull_model "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Неизвестная команда: $1"
            show_help
            exit 1
            ;;
    esac
}

# Запуск скрипта
main "$@"
