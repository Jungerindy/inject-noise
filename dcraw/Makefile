CC = gcc
CFLAGS = -O4
LIBS = -lm -DNODEPS

TARGET = dcraw
SRC = dcraw.c


all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LIBS) $(DEFS)

debug: CFLAGS = -g -O0
debug: all

patch:
	@echo "Applying patch"
	# patch < my_noise_injection.patch

clean:
	rm -f $(TARGET)

.PHONY: all clean debug patch