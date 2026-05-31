// 参考资料类型定义 + 数据加载

export interface ResourceItem {
  name: string
  url: string | null
  desc: string
  path?: string
  size?: string
}

export interface ResourceSubcategory {
  title: string
  items: ResourceItem[]
}

export interface ResourceCategory {
  id: string
  title: string
  icon: string
  subcategories: ResourceSubcategory[]
}

import defaultData from './resources.json'

export const RESOURCE_CATEGORIES: ResourceCategory[] = defaultData as ResourceCategory[]
