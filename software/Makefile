# Variables
CC = g++
CFLAGS = -Wall -Wextra -std=c++11

# Fichiers source et objets
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# Nom de l'exécutable
EXEC = main

# Règle par défaut
all: $(EXEC)

# Règle pour l'exécutable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $(EXEC) $(OBJS)

# Règle pour les objets
%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

# Nettoyage
clean:
	rm -f $(OBJS) $(EXEC)
