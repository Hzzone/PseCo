import torch
import tqdm


def dump_clip_text_features(classes):
    from ops.foundation_models import clip

    clip.available_models()

    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    # @title Define hyperparameters
    FLAGS = {
        'prompt_engineering': True,
        'this_is': True,

        'temperature': 100.0,
        'use_softmax': False,
    }
    from easydict import EasyDict
    FLAGS = EasyDict(FLAGS)

    def article(name):
        return 'an' if name[0] in 'aeiou' else 'a'

    def processed_name(name, rm_dot=False):
        # _ for lvis
        # / for obj365
        res = name.replace('_', ' ').replace('/', ' or ').lower()
        if rm_dot:
            res = res.rstrip('.')
        return res

    single_template = [
        'a photo of {article} {}.'
    ]

    multiple_templates = [
        'There is {article} {} in the scene.',
        'There is the {} in the scene.',
        'a photo of {article} {} in the scene.',
        'a photo of the {} in the scene.',
        'a photo of one {} in the scene.',

        'itap of {article} {}.',
        'itap of my {}.',  # itap: I took a picture of
        'itap of the {}.',
        'a photo of {article} {}.',
        'a photo of my {}.',
        'a photo of the {}.',
        'a photo of one {}.',
        'a photo of many {}.',

        'a good photo of {article} {}.',
        'a good photo of the {}.',
        'a bad photo of {article} {}.',
        'a bad photo of the {}.',
        'a photo of a nice {}.',
        'a photo of the nice {}.',
        'a photo of a cool {}.',
        'a photo of the cool {}.',
        'a photo of a weird {}.',
        'a photo of the weird {}.',

        'a photo of a small {}.',
        'a photo of the small {}.',
        'a photo of a large {}.',
        'a photo of the large {}.',

        'a photo of a clean {}.',
        'a photo of the clean {}.',
        'a photo of a dirty {}.',
        'a photo of the dirty {}.',

        'a bright photo of {article} {}.',
        'a bright photo of the {}.',
        'a dark photo of {article} {}.',
        'a dark photo of the {}.',

        'a photo of a hard to see {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of {article} {}.',
        'a low resolution photo of the {}.',
        'a cropped photo of {article} {}.',
        'a cropped photo of the {}.',
        'a close-up photo of {article} {}.',
        'a close-up photo of the {}.',
        'a jpeg corrupted photo of {article} {}.',
        'a jpeg corrupted photo of the {}.',
        'a blurry photo of {article} {}.',
        'a blurry photo of the {}.',
        'a pixelated photo of {article} {}.',
        'a pixelated photo of the {}.',

        'a black and white photo of the {}.',
        'a black and white photo of {article} {}.',

        'a plastic {}.',
        'the plastic {}.',

        'a toy {}.',
        'the toy {}.',
        'a plushie {}.',
        'the plushie {}.',
        'a cartoon {}.',
        'the cartoon {}.',

        'an embroidered {}.',
        'the embroidered {}.',

        'a painting of the {}.',
        'a painting of a {}.',
    ]

    def build_text_embedding(categories):
        if FLAGS.prompt_engineering:
            templates = multiple_templates
        else:
            templates = single_template

        with torch.no_grad():
            all_text_embeddings = {}
            print('Building text embeddings...')
            for category in tqdm.tqdm(categories):
                texts = [
                    template.format(processed_name(category, rm_dot=True),
                                    article=article(category))
                    for template in templates]
                if FLAGS.this_is:
                    texts = [
                        'This is ' + text if text.startswith('a') or text.startswith('the') else text
                        for text in texts
                    ]
                texts = clip.tokenize(texts)  # tokenize
                texts = texts.cuda()
                text_embeddings = model.encode_text(texts)  # embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                all_text_embeddings[category] = text_embedding.float().cpu()

        return all_text_embeddings

    # text CLIP embeddings
    imagenet_classes = classes
    zeroshot_weights = build_text_embedding(imagenet_classes)
    return zeroshot_weights


def dump_clip_text_features_detic(classes):
    from ops.foundation_models import clip

    clip.available_models()

    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    def build_text_embedding(categories):
        with torch.no_grad():
            all_text_embeddings = {}
            print('Building text embeddings...')
            for category in tqdm.tqdm(categories):
                texts = [f'a {category}']
                texts = clip.tokenize(texts)  # tokenize
                texts = texts.cuda()
                text_embeddings = model.encode_text(texts)  # embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                all_text_embeddings[category] = text_embedding.float().cpu()

        return all_text_embeddings

    # text CLIP embeddings
    imagenet_classes = classes
    zeroshot_weights = build_text_embedding(imagenet_classes)
    return zeroshot_weights


def dump_clip_image_features(image, bboxes):
    import torchvision
    from torchvision.transforms.functional import to_pil_image, to_tensor
    import torch.nn.functional as F

    normalize = torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                 std=(0.26862954, 0.26130258, 0.27577711))

    from ops.foundation_models import clip

    clip.available_models()

    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    examples = []
    for box in bboxes:
        example = image.crop(box.long().tolist())
        example = example.resize((224, 224))
        example = normalize(to_tensor(example)).unsqueeze(0)
        examples.append(example)
    examples = torch.cat(examples)
    e = []
    with torch.no_grad():
        for indices in torch.arange(len(examples)).split(256):
            e.append(model.encode_image(examples[indices].cuda()).float())
    e = torch.cat(e, dim=0)
    e = F.normalize(e, dim=1).cpu()
    return e
