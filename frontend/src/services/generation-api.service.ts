
import { Injectable, inject, signal } from '@angular/core';
import { VellumiumService } from './vellumium.service';
import { AuthService } from './auth.service';
import { SupabaseService } from './supabase.service';

const LOCAL_API = 'http://localhost:7860';

// ─── Generation Types ──────────────────────────────────────
export type GenerationMode = 't2i' | 'inpaint' | 'controlnet' | 'qwen_edit';
export type ControlNetType = 'canny' | 'hed' | 'depth' | 'pose' | 'mlsd';

export interface LoRASelection {
  name: string;
  filename: string;
  scale: number;
  enabled: boolean;
}

export interface GenerationParams {
  mode: GenerationMode;
  prompt: string;
  negative_prompt: string;
  model: 'turbo' | 'base';
  width: number;
  height: number;
  steps: number;
  seed: number;
  scheduler: string;
  guidance_scale: number;

  // LoRA
  loras: { filename: string; scale: number }[];

  // Inpaint
  mask_base64?: string;
  input_image_base64?: string;
  mask_blur?: number;
  denoise_strength?: number;

  // ControlNet
  controlnet_type?: ControlNetType;
  controlnet_scale?: number;
  control_image_base64?: string;

  // Qwen Edit
  edit_instruction?: string;
  source_images_base64?: string[];
  reference_images_base64?: string[];
  qwen_lora?: string;
}

export interface QueueItem {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result_url?: string;
  error?: string;
}

// ─── Resolution Presets ────────────────────────────────────
export const RESOLUTION_PRESETS = {
  '1MP': [
    { label: '1024 x 1024 (1:1)', w: 1024, h: 1024 },
    { label: '1280 x 768 (5:3)', w: 1280, h: 768 },
    { label: '768 x 1280 (3:5)', w: 768, h: 1280 },
    { label: '1536 x 640 (12:5)', w: 1536, h: 640 },
    { label: '640 x 1536 (5:12)', w: 640, h: 1536 },
    { label: '1152 x 896 (9:7)', w: 1152, h: 896 },
    { label: '896 x 1152 (7:9)', w: 896, h: 1152 },
  ],
  '2MP': [
    { label: '1440 x 1440 (1:1)', w: 1440, h: 1440 },
    { label: '1920 x 1088 (16:9)', w: 1920, h: 1088 },
    { label: '1088 x 1920 (9:16)', w: 1088, h: 1920 },
    { label: '1600 x 1280 (5:4)', w: 1600, h: 1280 },
    { label: '1280 x 1600 (4:5)', w: 1280, h: 1600 },
  ],
};

export const SCHEDULERS = [
  'Euler',
  'Euler + Beta',
];

@Injectable({
  providedIn: 'root'
})
export class GenerationApiService {
  private vellumService = inject(VellumiumService);
  private authService = inject(AuthService);
  private supabaseService = inject(SupabaseService);
  private supabase = this.supabaseService.supabase;

  readonly isSubmitting = signal(false);
  readonly lastError = signal<string | null>(null);
  readonly queueStatus = signal<QueueItem | null>(null);

  // Available LoRAs (populated from status endpoint)
  readonly availableLoRAs = signal<LoRASelection[]>([]);
  readonly availableControlNets = signal<string[]>([]);

  // ─── Fetch server status ─────────────────────────────────
  async fetchStatus() {
    try {
      const resp = await fetch(`${LOCAL_API}/status`);
      if (resp.ok) {
        const data = await resp.json();
        if (data.loras) {
          this.availableLoRAs.set(data.loras.map((l: any) => ({
            name: l.name || l.filename.replace('.safetensors', ''),
            filename: l.filename,
            scale: 1.0,
            enabled: false,
          })));
        }
        if (data.controlnets) {
          this.availableControlNets.set(data.controlnets);
        }
      }
    } catch {
      // Server not available — use empty lists
    }
  }

  // ─── Direct local generation (no queue) ─────────────────
  async generateDirect(params: GenerationParams): Promise<string | null> {
    this.isSubmitting.set(true);
    this.lastError.set(null);

    try {
      const body: Record<string, any> = {
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        model: params.model === 'turbo' ? 'z-image-turbo' : 'z-image-base',
        width: params.width,
        height: params.height,
        steps: params.steps,
        seed: params.seed === -1 ? Math.floor(Math.random() * 2147483647) : params.seed,
        scheduler: params.scheduler,
        guidance_scale: params.guidance_scale,
        mode: params.mode,
      };

      if (params.loras.length > 0) {
        body['loras'] = params.loras.map(l => ({ [l.filename]: l.scale }));
      }

      if (params.mode === 'inpaint') {
        body['input_image_base64'] = params.input_image_base64;
        body['mask_base64'] = params.mask_base64;
        body['mask_blur'] = params.mask_blur ?? 12;
        body['denoise_strength'] = params.denoise_strength ?? 1.0;
      }

      if (params.mode === 'controlnet') {
        body['controlnet_type'] = params.controlnet_type;
        body['controlnet_scale'] = params.controlnet_scale ?? 0.75;
        body['control_image_base64'] = params.control_image_base64;
      }

      const resp = await fetch(`${LOCAL_API}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `Generation failed: ${resp.status}`);
      }

      const data = await resp.json();
      return data.url || null;
    } catch (e: any) {
      this.lastError.set(e.message || 'Local generation failed');
      return null;
    } finally {
      this.isSubmitting.set(false);
    }
  }

  // ─── Submit to Supabase queue (Z-Image) ──────────────────
  async submitToQueue(sceneId: string, layerId: string, params: GenerationParams): Promise<boolean> {
    const user = this.authService.currentUser();
    if (!user) return false;

    this.isSubmitting.set(true);
    this.lastError.set(null);

    try {
      const queueParams: Record<string, any> = {
        mode: params.mode,
        model: params.model,
        width: params.width,
        height: params.height,
        steps: params.steps,
        seed: params.seed === -1 ? Math.floor(Math.random() * 2147483647) : params.seed,
        scheduler: params.scheduler,
        guidance_scale: params.guidance_scale,
        negative_prompt: params.negative_prompt,
      };

      // LoRAs
      if (params.loras.length > 0) {
        queueParams['loras'] = params.loras;
      }

      // Inpaint
      if (params.mode === 'inpaint') {
        queueParams['mask_base64'] = params.mask_base64;
        queueParams['input_image_base64'] = params.input_image_base64;
        queueParams['mask_blur'] = params.mask_blur ?? 8;
        queueParams['denoise_strength'] = params.denoise_strength ?? 0.75;
      }

      // ControlNet
      if (params.mode === 'controlnet') {
        queueParams['controlnet_type'] = params.controlnet_type;
        queueParams['controlnet_scale'] = params.controlnet_scale ?? 0.75;
        queueParams['control_image_base64'] = params.control_image_base64;
      }

      const { error } = await this.supabase
        .from('generation_queue')
        .insert({
          user_id: user.id,
          scene_id: sceneId,
          layer_id: layerId,
          prompt: params.prompt,
          params: queueParams,
          status: 'pending',
        });

      if (error) throw error;
      return true;
    } catch (e: any) {
      this.lastError.set(e.message || 'Failed to submit to queue');
      return false;
    } finally {
      this.isSubmitting.set(false);
    }
  }

  // ─── Qwen Image Edit (direct) ────────────────────────────
  async qwenEdit(params: {
    instruction: string;
    source_images: string[];
    reference_images?: string[];
    lora?: string;
  }): Promise<string | null> {
    this.isSubmitting.set(true);
    this.lastError.set(null);

    try {
      const resp = await fetch('http://localhost:8200/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instruction: params.instruction,
          source_images: params.source_images,
          reference_images: params.reference_images || [],
          lora: params.lora || null,
        }),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.error || `Qwen Edit failed: ${resp.status}`);
      }

      const data = await resp.json();
      return data.url || null;
    } catch (e: any) {
      this.lastError.set(e.message || 'Qwen Edit request failed');
      return null;
    } finally {
      this.isSubmitting.set(false);
    }
  }

  // ─── Subscribe to queue updates ──────────────────────────
  subscribeToQueue(sceneId: string, callback: (item: QueueItem) => void): () => void {
    const channel = this.supabase
      .channel(`queue_${sceneId}`)
      .on(
        'postgres_changes' as any,
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'generation_queue',
          filter: `scene_id=eq.${sceneId}`,
        },
        (payload: any) => {
          const row = payload.new;
          callback({
            id: row.id,
            status: row.status,
            result_url: row.result_url,
            error: row.error,
          });
        }
      )
      .subscribe();

    return () => {
      this.supabase.removeChannel(channel);
    };
  }
}
