# Cache-Control 设置

```html
Cache-Control: max-age=2592000, stale-while-revalidate=600, stale-if-error=86400
```

* max-age=2592000	30 天（2592000 秒）	内容在 30 天内被视为“新鲜”，CloudFront 可直接返回，无需回源
* stale-while-revalidate=600	10 分钟	过期后 10 分钟内，CloudFront 仍可返回旧内容，同时后台异步回源更新（提升用户体验）
* stale-if-error=86400	24 小时	若回源失败（如源站宕机），CloudFront 可继续提供已过期但缓存的内容长达 24 小时