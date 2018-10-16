#pragma once

#include <vector>
#include <algorithm>
#include <iterator>
#include <functional>

namespace aten
{
template <class Data>
class DataList {
public:
    DataList() {}
    ~DataList() {}

private:
    DataList(const DataList& rhs) = delete;
    const DataList& operator=(const DataList& rhs) = delete;

public:
    class ListItem {
        friend class DataList<Data>;

    public:
        ListItem() {}
        ~ListItem()
        {
            leave();
        }

    private:
        ListItem(const ListItem& rhs) = delete;
        const ListItem& operator=(const ListItem& rhs) = delete;

    public:
        using FuncWhenAnyLeave = std::function<void(Data*)>;

        void init(Data* data, FuncWhenAnyLeave func)
        {
            m_data = data;
            m_funcWhenAnyLeave = func;
        }

        bool leave()
        {
            bool ret = false;

            if (m_belongedList) {
                auto list = m_belongedList;
                ret = list->remove(*this);

                if (ret && m_funcWhenAnyLeave) {
                    auto& realList = list->m_list;

                    for each (auto& item in realList)
                    {
                        auto data = item->m_data;
                        m_funcWhenAnyLeave(data);
                    }
                }
            }

            return ret;
        }

        Data* getData()
        {
            return m_data;
        }

        const Data* getData() const
        {
            return m_data;
        }

        int currentIndex() const
        {
            int idx = -1;

            if (m_belongedList) {
                idx = m_belongedList->currentIndex(*this);
            }

            return idx;
        }

    private:
        Data* m_data{ nullptr };
        DataList<Data>* m_belongedList{ nullptr };
        FuncWhenAnyLeave m_funcWhenAnyLeave{ nullptr };
    };

    void add(ListItem& item)
    {
        item.leave();
        item.m_blongedList = this;

        m_list.push_back(item);
    }

    bool remove(ListItem& item)
    {
        if (item.m_belongedList != this) {
            AT_ASSERT(false);
            return false;
        }

        auto it = std::find(m_list.begin(), m_list.end(), &item);
            
        if (it != m_list.end()) {
            m_list.erase(it);
                
            item.m_belongedList = nullptr;

            return true;
        }
        return false;
    }

    int currentIndex(const ListItem& item)
    {
        if (item.m_belongedList != this) {
            AT_ASSERT(false);
            return -1;
        }

        auto it = std::find(m_list.begin(), m_list.end(), &item);

        if (it != m_list.end()) {
            auto ret = std::distance(m_list.begin(), it);
            return static_cast<int>(ret);
        }

        return -1;
    }

    uint32_t size() const
    {
        return static_cast<uint32_t>(m_list.size());
    }

    Data* operator[](int idx)
    {
        return m_list[idx]->m_data;
    }

    const Data* operator[](int idx) const
    {
        return m_list[idx]->m_data;
    }

    const std::vector<ListItem*>& getList() const
    {
        return m_list;
    }

private:
    std::vector<ListItem*> m_list;
};
}
