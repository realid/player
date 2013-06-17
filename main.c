#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/avstring.h>
#include <SDL/SDL.h>

#define SDL_AUDIO_BUFFER_SIZE       1024
#define MAX_AUDIOQ_SIZE             (1 * 1024 * 1024)
#define MAX_VIDEOQ_SIZE             (1 * 1024 * 1024)
#define FF_ALLOC_EVENT              (SDL_USEREVENT)
#define FF_REFRESH_EVENT            (SDL_USEREVENT + 1)
#define FF_QUIT_EVENT               (SDL_USEREVENT + 2)
#define VIDEO_PICTURE_QUEUE_SIZE    1

static int sws_flags = SWS_BICUBIC;

typedef struct PacketQueue
{
    AVPacketList *first_pkt, *last_pkt;
    int nb_packets;
    int size;
    SDL_mutex *mutex; // 互斥锁
    SDL_cond *cond; // 条件变量，当list为空时，线程等待
} PacketQueue;

typedef struct VideoPicture {
    SDL_Overlay *bmp;
    int width, height;
    int allocated;
    double pts;
} VideoPicture;

typedef struct VideoState {
    AVFormatContext     *ic;

    int                 audioStream;
    double              audio_clock;
    AVStream            *audio_st;
    AVFrame             *audio_frame;
    PacketQueue         audioq;
    unsigned int        audio_buf_size;
    unsigned int        audio_buf_index;
    AVPacket            audio_pkt;
    uint8_t             *audio_pkt_data;
    int                 audio_pkt_size;
    uint8_t             *audio_buf;
    uint8_t             *audio_buf1;
    DECLARE_ALIGNED(16, uint8_t, audio_buf2)[AVCODEC_MAX_AUDIO_FRAME_SIZE * 4];
    enum AVSampleFormat audio_src_fmt;
    enum AVSampleFormat audio_tgt_fmt;
    int                 audio_src_channels;
    int                 audio_tgt_channels;
    int64_t             audio_src_channel_layout;
    int64_t             audio_tgt_channel_layout;
    int                 audio_src_freq;
    int                 audio_tgt_freq;
    struct SwrContext   *swr_ctx;

    int                 videoStream;
    AVStream            *video_st;
    AVFrame             *video_frame;
    PacketQueue         videoq;
    
    VideoPicture        pictq[VIDEO_PICTURE_QUEUE_SIZE];
    int                 pictq_size, pictq_rindex, pictq_windex;
    SDL_mutex           *pictq_mutex;
    SDL_cond            *pictq_cond;

    SDL_Thread          *parse_tid;
    SDL_Thread          *video_tid;

    char                filename[1024];
    int                 quit;
} VideoState;

VideoState *global_video_state;

SDL_Surface *screen;

void packet_queue_init(PacketQueue *q)
{
    memset(q, 0, sizeof(PacketQueue));
    q->mutex = SDL_CreateMutex();
    q->cond = SDL_CreateCond();
}

int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
    AVPacketList *pktl;
    pktl = (AVPacketList *)av_malloc(sizeof(AVPacketList));
    if (!pktl)
        return -1;
    pktl->pkt = *pkt;
    pktl->next = NULL;
    SDL_LockMutex(q->mutex);
    if (!q->last_pkt) //当队列为空时
        q->first_pkt = pktl;
    else
        q->last_pkt->next = pktl;
    q->last_pkt = pktl;
    q->nb_packets++;
    q->size += pktl->pkt.size;
    SDL_CondSignal(q->cond);
    SDL_UnlockMutex(q->mutex);

    return 0;
}

static int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
    AVPacketList *pktl;
    int ret;

    SDL_LockMutex(q->mutex);
    for (;;) {
        if (global_video_state->quit) {
            ret = -1;
            break;
        }
        pktl = q->first_pkt;
        if (pktl) {
            q->first_pkt = pktl->next;
            if (!q->first_pkt)
                q->last_pkt = NULL;
            q->nb_packets--;
            q->size -= pktl->pkt.size;
            *pkt = pktl->pkt;
            av_free(pktl);
            ret = 1;
            break;
        } else if (!block) {
            ret = 0;
            break;
        } else
            SDL_CondWait(q->cond, q->mutex);
    }
    SDL_UnlockMutex(q->mutex);
    return ret;
}

int audio_decode_frame(VideoState *is)
{
    AVPacket *pkt = &is->audio_pkt;
    int got_frame = 0;
    int len1, len2, decoded_data_size;
    uint64_t dec_channel_layout;
    int wanted_nb_samples, resampled_data_size;

    for (;;) {
        while (is->audio_pkt_size > 0) {
            if (!is->audio_frame) {
                if (!(is->audio_frame = avcodec_alloc_frame())) {
                    fprintf(stderr, "avcodec_alloc_frame failed\n");
                    return -1;
                }
            } else
                avcodec_get_frame_defaults(is->audio_frame);

            len1 = avcodec_decode_audio4(is->audio_st->codec, is->audio_frame, &got_frame, pkt);
            if (len1 < 0) {
                // error, skip the frame
                is->audio_pkt_size = 0;
                break;
            }
            is->audio_pkt_data += len1;
            is->audio_pkt_size -= len1;
            if (!got_frame)
                continue;

            /* 计算解码出来的帧需要的缓存大小 */
            decoded_data_size = av_samples_get_buffer_size(NULL,
                    is->audio_frame->channels,
                    is->audio_frame->nb_samples,
                    is->audio_frame->format, 1);

//            return decoded_data_size;
            dec_channel_layout = (is->audio_frame->channel_layout && is->audio_frame->channels 
                    == av_get_channel_layout_nb_channels(is->audio_frame->channel_layout))
                    ? is->audio_frame->channel_layout 
                    : av_get_default_channel_layout(is->audio_frame->channels);
            wanted_nb_samples = is->audio_frame->nb_samples;

            if (is->audio_frame->format != is->audio_src_fmt ||
                    dec_channel_layout != is->audio_src_channel_layout ||
                    is->audio_frame->sample_rate != is->audio_src_freq ||
                    (wanted_nb_samples != is->audio_frame->nb_samples && !is->swr_ctx)) {
                if (is->swr_ctx)
                    swr_free(&is->swr_ctx);
                is->swr_ctx = swr_alloc_set_opts(NULL,
                        is->audio_tgt_channel_layout,
                        is->audio_tgt_fmt,
                        is->audio_tgt_freq,
                        dec_channel_layout,
                        is->audio_frame->format,
                        is->audio_frame->sample_rate,
                        0, NULL);
                if (!is->swr_ctx || swr_init(is->swr_ctx) < 0) {
                    fprintf(stderr, "swr_init() failed\n");
                    break;
                }
                is->audio_src_channel_layout = dec_channel_layout;
                is->audio_src_channels = is->audio_st->codec->channels;
                is->audio_src_freq = is->audio_st->codec->sample_rate;
                is->audio_src_fmt = is->audio_st->codec->sample_fmt;
            }
            if (is->swr_ctx) {
                const uint8_t **in = (const uint8_t **)is->audio_frame->extended_data;
                uint8_t *out[] = {is->audio_buf2};
                if (wanted_nb_samples != is->audio_frame->nb_samples) {
                    if (swr_set_compensation(is->swr_ctx, 
                                (wanted_nb_samples - is->audio_frame->nb_samples) 
                                * is->audio_tgt_freq / is->audio_frame->sample_rate,
                                wanted_nb_samples 
                                * is->audio_tgt_freq / is->audio_frame->sample_rate) < 0){
                        fprintf(stderr, "swr_set_compensation() failed\n");
                        break;
                    }
                }
                len2 = swr_convert(is->swr_ctx, out, sizeof(is->audio_buf2)
                        / is->audio_tgt_channels
                        / av_get_bytes_per_sample(is->audio_tgt_fmt),
                        in, is->audio_frame->nb_samples);
                if (len2 < 0) {
                    fprintf(stderr, "swr_convert() failed\n");
                    break;
                }
                if (len2 == sizeof(is->audio_buf2) / is->audio_tgt_channels / av_get_bytes_per_sample(is->audio_tgt_fmt)) {
                    fprintf(stderr, "warning: audio buffer is probably too small\n");
                    swr_init(is->swr_ctx);
                }
                is->audio_buf = is->audio_buf2;
                resampled_data_size = len2 * is->audio_tgt_channels * av_get_bytes_per_sample(is->audio_tgt_fmt);
            } else {
                resampled_data_size = decoded_data_size;
                is->audio_buf = is->audio_frame->data[0];
            }
            return resampled_data_size;
        }

        if (pkt->data)
            av_free_packet(pkt);
        memset(pkt, 0, sizeof(*pkt));
        if (is->quit)
            return -1;
        if (packet_queue_get(&is->audioq, pkt, 1) < 0) {
            fprintf(stderr, "packet_queue_get: failed\n");
            return -1;
        }
        is->audio_pkt_data = pkt->data;
        is->audio_pkt_size = pkt->size;
    }
}

void audio_callback(void *userdata, Uint8 *stream, int len)
{
    VideoState *is = (VideoState*)userdata;
    int len1, audio_data_size;
    /* len是SDL传入的SDL缓冲区大小，如果这个缓冲未满，一直填充数据*/
    while (len > 0) {
        /* audio_buf_index和audio_buf_size标示我们自己用来放置解码出来的数据的缓冲区  */
        /* 这些数据待copy到SDL缓冲区，当audio_buf_index >= audio_buf_size时，意味着我 */
        /* 们的缓冲为空，没有数据可供copy，这时候需要调用audio_decode_frame来解码更多 */
        /* 的帧数据 */
        if (is->audio_buf_index >= is->audio_buf_size) {
            audio_data_size = audio_decode_frame(is);
            if (audio_data_size < 0) {
                is->audio_buf_size = 1024;
                memset(is->audio_buf, 0, is->audio_buf_size);
            } else
                is->audio_buf_size = audio_data_size;
            is->audio_buf_index = 0;
        } 
        len1 = is->audio_buf_size - is->audio_buf_index;
        if (len1 > len)
            len1 = len;
        memcpy(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1);
        len -= len1;
        stream += len1;
        is->audio_buf_index += len1;
    }
}

/* 根据定时器的设定，发送FF_REFRESH_EVENT事件 */
static uint32_t sdl_refresh_timer_cb(uint32_t interval, void *opaque)
{
    SDL_Event event;
    event.type = FF_REFRESH_EVENT;
    event.user.data1 = opaque;
    SDL_PushEvent(&event);
    
    return 0;
}

/* 设置定时器，延时delay调用sdl_refresh_timer_cb()一次 */
static void schedule_refresh(VideoState *is, int delay)
{
    SDL_AddTimer(delay, sdl_refresh_timer_cb, is);
}

void video_display(VideoState *is)
{
    SDL_Rect rect;
    VideoPicture *vp;
    AVPicture pict;
    float aspect_ratio;
    int w, h, x, y;
    int i;

    vp = &is->pictq[is->pictq_rindex];
    if (vp->bmp) {
        /* 获取视频纵横比例 */
        if (is->video_st->codec->sample_aspect_ratio.num == 0) // unknown
            aspect_ratio = 0;
        else
            aspect_ratio = av_q2d(is->video_st->codec->sample_aspect_ratio) *
                is->video_st->codec->width /  is->video_st->codec->height;
        if (aspect_ratio <= 0.0)
            aspect_ratio = (float)is->video_st->codec->width / (float)is->video_st->codec->height;
        h = screen->h;
        w = ((int)(h * aspect_ratio)) & -3;
        if (w > screen->w) {
            w = screen->w;
            h = ((int)(w / aspect_ratio)) & -3;
        }
        x = (screen->w - w) / 2;
        y = (screen->h - h) / 2;

        rect.x = x;
        rect.y = y;
        rect.w = w;
        rect.h = h;
        SDL_DisplayYUVOverlay(vp->bmp, &rect);
    }
}


/* 播放视频的每一帧，循环调用schedule_refresh播放 */
void video_refresh_timer(void *userdata)
{
    VideoState *is = (VideoState *)userdata;
    VideoPicture *vp;

    if (is->video_st) {
        if (is->pictq_size == 0)
            schedule_refresh(is, 1);
        else {
            vp = &is->pictq[is->pictq_rindex];
            schedule_refresh(is, 20);
            video_display(is);

            if (++is->pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE)
                is->pictq_rindex = 0;
            SDL_LockMutex(is->pictq_mutex);
            is->pictq_size--;
            SDL_CondSignal(is->pictq_cond);
            SDL_UnlockMutex(is->pictq_mutex);
        }
    } else
        schedule_refresh(is, 100);
}


/* 创建用于显示的SDL_Overlay */
void alloc_picture(void *userdata)
{
    VideoState *is = (VideoState *)userdata;
    VideoPicture *vp;

    vp = &is->pictq[is->pictq_windex];
    if (vp->bmp)
        SDL_FreeYUVOverlay(vp->bmp);

    vp->width = is->video_st->codec->width;
    vp->height = is->video_st->codec->height;
    vp->bmp = SDL_CreateYUVOverlay(vp->width, vp->height, SDL_IYUV_OVERLAY, screen);

    SDL_LockMutex(is->pictq_mutex);
    vp->allocated = 1;
    SDL_CondSignal(is->pictq_cond);
    SDL_UnlockMutex(is->pictq_mutex);
}

int queue_picture(VideoState *is, AVFrame *pFrame)
{
    VideoPicture *vp;
    int dst_pix_fmt;
    AVPicture pict;
    static struct SwsContext *img_convert_ctx;
    /* 设置视频转换参数 */
    if (img_convert_ctx == NULL) {
        img_convert_ctx = sws_getContext(is->video_st->codec->width, is->video_st->codec->height,
                is->video_st->codec->pix_fmt,
                is->video_st->codec->width, is->video_st->codec->height,
                PIX_FMT_YUV420P,
                sws_flags, NULL, NULL, NULL);
        if (img_convert_ctx == NULL) {
            fprintf(stderr, "Cannot initialize the conversion context\n");
            exit(1);
        }
    }

    SDL_LockMutex(is->pictq_mutex);
    while (is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE && !is->quit)
        SDL_CondWait(is->pictq_cond, is->pictq_mutex);
    SDL_UnlockMutex(is->pictq_mutex);

    if (is->quit)
        return -1;

    vp = &is->pictq[is->pictq_windex];

    /* 如果is->pictq为空，发送FF_ALLOC_EVENT等待分配空间 */
    if (!vp->bmp || vp->width != is->video_st->codec->width || vp->height != is->video_st->codec->height) {
        SDL_Event event;
        vp->allocated = 0;
        event.type = FF_ALLOC_EVENT;
        event.user.data1 = is;
        SDL_PushEvent(&event);

        SDL_LockMutex(is->pictq_mutex);
        while (!vp->allocated && !is->quit)
            SDL_CondWait(is->pictq_cond, is->pictq_mutex);
        SDL_UnlockMutex(is->pictq_mutex);
        if (is->quit)
            return -1;
    }

    /* 如果is->pictq不为空，做好显示帧的准备工作 */
    if (vp->bmp) {
        SDL_LockYUVOverlay(vp->bmp);
        dst_pix_fmt = PIX_FMT_YUV420P;
        pict.data[0] = vp->bmp->pixels[0];
        pict.data[1] = vp->bmp->pixels[1];
        pict.data[2] = vp->bmp->pixels[2];
        pict.linesize[0] = vp->bmp->pitches[0];
        pict.linesize[1] = vp->bmp->pitches[1];
        pict.linesize[2] = vp->bmp->pitches[2];

        /* 视频帧转换 */
        sws_scale(img_convert_ctx, (const uint8_t * const *)pFrame->data, pFrame->linesize,
                0, vp->height, pict.data, pict.linesize);
        SDL_UnlockYUVOverlay(vp->bmp);
        if (++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE)
            is->pictq_windex = 0;
        SDL_LockMutex(is->pictq_mutex);
        is->pictq_size++;
        SDL_UnlockMutex(is->pictq_mutex);
    }
    return 0;
}

int video_thread(void *arg)
{
    VideoState *is = (VideoState *)arg;
    AVPacket pkt1, *packet = &pkt1;
    int len1, frameFinished;
    AVFrame *pFrame;

    pFrame = avcodec_alloc_frame();

    for (;;) {
        if (packet_queue_get(&is->videoq, packet, 1) < 0) {
            fprintf(stderr, "packet_queue_get: failed\n");
            break;
        }

        len1 = avcodec_decode_video2(is->video_st->codec, pFrame, &frameFinished, packet);
        if (frameFinished)
            /* 当一帧取完后，将该帧加入is->pictq队列中 */
            if (queue_picture(is, pFrame) < 0)
                break;
        av_free_packet(packet);
    }
    av_free(pFrame);
    return 0;
}

int stream_component_open(VideoState *is, int stream_index)
{
    AVFormatContext *ic = is->ic;
    AVCodecContext *codecCtx;
    AVCodec *codec;
    /* wanted_spec为期望设置的参数，spec是系统最终接受的参数 */
    /* 需要检查系统接受的参数是否正确 */
    SDL_AudioSpec wanted_spec, spec;
    int64_t wanted_channel_layout = 0; // 声道布局
    int wanted_nb_channels; // 声道数
    /* SDL支持的声道数为1, 2, 4, 6 */
    /* 该数组用来纠正不支持的声道数目 */
    const int next_nb_channels[] = {0, 0, 1, 6, 2, 6, 4, 6};

    if (stream_index < 0 || stream_index >= ic->nb_streams)
        return -1;
    codecCtx = ic->streams[stream_index]->codec;
    /* 初始化声音设备 */
    if (codecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
        wanted_nb_channels = codecCtx->channels;
        if (!wanted_channel_layout || wanted_nb_channels != av_get_channel_layout_nb_channels(wanted_channel_layout)) {
            wanted_channel_layout = av_get_default_channel_layout(wanted_nb_channels);
            wanted_channel_layout &= ~AV_CH_LAYOUT_STEREO_DOWNMIX;
        }
        wanted_spec.channels = av_get_channel_layout_nb_channels(wanted_channel_layout);
        wanted_spec.freq = codecCtx->sample_rate;
        if (wanted_spec.freq <= 0 || wanted_spec.channels <= 0) {
            fprintf(stderr, "Invalid sample rate or channel count!\n");
            return -1;
        }
        wanted_spec.format = AUDIO_S16SYS;
        wanted_spec.silence = 0; // 0指示静音
        wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;
        wanted_spec.callback = audio_callback;
        wanted_spec.userdata = is;
        while (SDL_OpenAudio(&wanted_spec, &spec) < 0) {
            fprintf(stderr, "SDL_OpenAudio(%d channels): %s\n", wanted_spec.channels, SDL_GetError());
            wanted_spec.channels = next_nb_channels[FFMIN(7, wanted_spec.channels)];
            if (!wanted_spec.channels) {
                fprintf(stderr, "No more channel to try\n");
                return -1;
            }
            wanted_channel_layout = av_get_default_channel_layout(wanted_spec.channels);
        }
        if (spec.format != AUDIO_S16SYS) {
            fprintf(stderr, "SDL advised audio format %d is not supported\n", spec.format);
            return -1;
        }
        if (spec.channels != wanted_spec.channels) {
            wanted_channel_layout = av_get_default_channel_layout(spec.channels);
            if (!wanted_channel_layout) {
                fprintf(stderr, "SDL advised channel count %d is not supported\n", spec.channels);
                return -1;
            }
        }
        /*
        fprintf(stderr, "%d: wanted_spec.format = %d\n", __LINE__, wanted_spec.format);
        fprintf(stderr, "%d: wanted_spec.samples = %d\n", __LINE__, wanted_spec.samples);
        fprintf(stderr, "%d: wanted_spec.channels = %d\n", __LINE__, wanted_spec.channels);
        fprintf(stderr, "%d: wanted_spec.freq = %d\n", __LINE__, wanted_spec.freq);

        fprintf(stderr, "%d: spec.format = %d\n", __LINE__, spec.format);
        fprintf(stderr, "%d: spec.samples = %d\n", __LINE__, spec.samples);
        fprintf(stderr, "%d: spec.channels = %d\n", __LINE__, spec.channels);
        fprintf(stderr, "%d: spec.freq = %d\n", __LINE__, spec.freq);
        */

        is->audio_src_fmt = is->audio_tgt_fmt = AV_SAMPLE_FMT_S16;
        is->audio_src_freq = is->audio_tgt_freq = spec.freq;
        is->audio_src_channel_layout = is->audio_tgt_channel_layout = wanted_channel_layout;
        is->audio_src_channels = is->audio_tgt_channels = spec.channels;
    }

    codec = avcodec_find_decoder(codecCtx->codec_id);
    if (!codec) {
        fprintf(stderr, "Unsupported codec!\n");
        return -1;
    }
    if (avcodec_open2(codecCtx, codec, NULL) < 0) {
        fprintf(stderr, "avcodec_open2\n");
        return -1;
    }
    ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;
    switch (codecCtx->codec_type) {
        case AVMEDIA_TYPE_AUDIO:
            is->audioStream = stream_index;
            is->audio_st = ic->streams[is->audioStream];
            is->audio_buf_size = 0;
            is->audio_buf_index = 0;
            memset(&is->audio_pkt, 0, sizeof(is->audio_pkt));
            packet_queue_init(&is->audioq);
            SDL_PauseAudio(0);
            break;
        case AVMEDIA_TYPE_VIDEO:
            is->videoStream = stream_index;
            is->video_st = ic->streams[is->videoStream];
            packet_queue_init(&is->videoq);
            is->video_tid = SDL_CreateThread(video_thread, is);
            break;
        default:
            break;
    }
    return 0;
}

int decode_interrupt_cb()
{
    return (global_video_state && global_video_state->quit);
}

static int decode_thread(void *arg)
{
    VideoState *is = (VideoState *)arg;
    AVFormatContext *ic = NULL;
    AVPacket pkt1, *packet = &pkt1;
    int i, ret, audio_index = -1, video_index = -1;
    static const AVIOInterruptCB int_cb = {decode_interrupt_cb, NULL};

    is->audioStream = -1;
    is->videoStream = -1;

    global_video_state = is;
    if (avformat_open_input(&ic, is->filename, NULL, NULL) != 0) {
        fprintf(stderr, "avformat_open_input failed\n");
        return -1;
    }
    ic->interrupt_callback = int_cb;
    is->ic = ic;
    if (avformat_find_stream_info(ic, NULL) < 0) {
        fprintf(stderr, "avformat_find_stream_info\n");
        return -1;
    }

    av_dump_format(ic, 0, is->filename, 0);
    for (i=0; i<ic->nb_streams; i++) {
        if (ic->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO && audio_index < 0)
            audio_index = i;
        if (ic->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO && video_index < 0)
            video_index = i;
    }

    if (audio_index >= 0)
        stream_component_open(is, audio_index);

    if (video_index >= 0)
        stream_component_open(is, video_index);

    if (is->audioStream < 0 && is->videoStream < 0) {
        fprintf(stderr, "%s: could not open codecs\n", is->filename);
        goto fail;
    }

    /* 循环读取packet，如果是video，则加到video队列中，如果是audio，则加到audio队列中 */
    for (;;) {
        if (is->quit)
            break;
        /* 当音频/视频队列已满，等待10ms */
        if (is->audioq.size > MAX_AUDIOQ_SIZE || is->videoq.size > MAX_VIDEOQ_SIZE) {
            SDL_Delay(10);
            continue;
        }
        /* 当ret为0时，表示读取正常，当ret<0时，表示错误或者文件读取完毕 */
        ret = av_read_frame(is->ic, packet);
        if (ret < 0) {
            /* 当媒体文件读取完毕时 AVERROR_xxx定义在libavutil/error.h中 */
            if (ret == AVERROR_EOF || url_feof(is->ic->pb))
                break;
            /* 当遇到读取错误时 */
            if (is->ic->pb && is->ic->pb->error)
                break;
            continue;
        }
        /* 音频/视频packet加入队列 */
        if (packet->stream_index == is->videoStream)
            packet_queue_put(&is->videoq, packet);
        else if (packet->stream_index == is->audioStream)
            packet_queue_put(&is->audioq, packet);
        else
            av_free_packet(packet);
    }

    /* 媒体读取完毕或遇到读取错误时，等待，当is->quit设置为1时，退出 */
    while (!is->quit)
        SDL_Delay(100);

fail: {
        SDL_Event event;
        event.type = FF_QUIT_EVENT;
        event.user.data1 = is;
        SDL_PushEvent(&event);
      }

    return 0;
}

int main(int argc, char **argv)
{
    SDL_Event event;
    VideoState *is;

    is = (VideoState *)av_mallocz(sizeof(VideoState));

    if (argc != 2) {
        printf("usage: %s <filename>\n", argv[0]);
        return -1;
    }

    av_register_all();
    if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_TIMER)) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return -1;
    }

    screen = SDL_SetVideoMode(640, 480, 24, 0);

    av_strlcpy(is->filename, argv[1], sizeof(is->filename));
    is->pictq_mutex = SDL_CreateMutex();
    is->pictq_cond = SDL_CreateCond();

    schedule_refresh(is, 40);
    is->parse_tid = SDL_CreateThread(decode_thread, is);
    if (!is->parse_tid) {
        av_free(is);
        return -1;
    }

    for (;;) {
        SDL_WaitEvent(&event);
        switch(event.type) {
            case FF_QUIT_EVENT:
                printf("FF_QUIT_EVENT recieved\n");
            case SDL_QUIT:
                printf("SDL_QUIT recieved\n");
                is->quit = 1; // 通知线程退出，否则线程将一直Delay
                SDL_Quit();
                return 0;
                break;
            case FF_ALLOC_EVENT:
                alloc_picture(event.user.data1);
                break;
            case FF_REFRESH_EVENT: // FF_REFRESH_EVENT事件的处理
                video_refresh_timer(event.user.data1); // 刷新视频画面
                break;
            default:
                break;
        }
    }
    return 0;
}
