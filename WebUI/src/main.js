// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from "vue"
import App from "./App"
import router from "./router"
import Vuetify from "vuetify"
import "vuetify/dist/vuetify.min.css"
import moment from "moment"

Vue.use(Vuetify)
Vue.directive('moment-ago', {
    bind(el, bindings) {
        const timestamp = bindings.value;
        el.innerHTML = moment(timestamp).fromNow()

        el.interval = setInterval(() => {
            const str = moment(timestamp).fromNow()
            if (el.innerHTML != str) {
                el.innerHTML = str
            }
        }, 10000)
    },

    unbind(el) {
        clearInterval(el.interval)
    }
})

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: "#app",
  router,
  components: { App },
  template: "<App/>"
})
