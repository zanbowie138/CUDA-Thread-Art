#pragma once
#include <SDL/SDL.h>

#define MAX_DISPLAY_SIZE 800

class ImageViewer
{
public:
	ImageViewer(const char* window_name, const std::vector<unsigned char> frame, const unsigned int size);
private:
	SDL_Window* window;
	SDL_Renderer* renderer;
};

inline ImageViewer::ImageViewer(const char* window_name, const std::vector<unsigned char> frame, const unsigned int size)
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		SDL_Log("SDL_Init failed: %s", SDL_GetError());
		return;
	}

	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "2");

	window = SDL_CreateWindow(window_name, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
	if (window == nullptr) {
		SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
		return;
	}

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (renderer == nullptr) {
		SDL_Log("SDL_CreateRenderer failed: %s", SDL_GetError());
		return;
	}

	SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STATIC, size, size);
	if (texture == nullptr) {
		SDL_Log("SDL_CreateTexture failed: %s", SDL_GetError());
		return;
	}

	SDL_UpdateTexture(texture, nullptr, utils::convert1c3c(frame.data(), size, size).data(), size * 3 * sizeof(unsigned char));
	SDL_RenderCopy(renderer, texture, nullptr, nullptr);
	SDL_RenderPresent(renderer);

	SDL_Event event;
	bool running = true;
	while (running) {
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				running = false;
			}
		}
		SDL_RenderCopy(renderer, texture, nullptr, nullptr);
		SDL_RenderPresent(renderer);
	}

	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
}