LIBS=-L/opt/local/lib -lSDLmain -Wl,-framework,AppKit -lavformat -lavcodec -lswscale -lavutil -lswresample -lSDL -Wl,-framework,Cocoa
CFLAGS=-I/opt/local/include -D_GNU_SOURCE=1 -D_THREAD_SAFE

player: main.c
	cc -g -o player $^ $(CFLAGS) $(LIBS)

convert: convert.c
	cc -g -o convert $^ $(CFLAGS) $(LIBS)

clean:
	rm -rf convert player
