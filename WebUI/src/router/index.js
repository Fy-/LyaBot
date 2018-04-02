import Vue from 'vue'
import Router from 'vue-router'
import ChatWindow from '@/components/ChatWindow'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Chat',
      component: ChatWindow 
    }
  ]
})
