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
                ret = m_belongedList->remove(this);
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
                idx = m_belongedList->currentIndex(this);
            }

            return idx;
        }

        Data* operator->()
        {
            return m_data;
        }

    private:
        void doFuncWhenAnyLeave()
        {
            if (m_funcWhenAnyLeave) {
                m_funcWhenAnyLeave(m_data);
            }
        }

    private:
        Data* m_data{ nullptr };
        DataList<Data>* m_belongedList{ nullptr };
        FuncWhenAnyLeave m_funcWhenAnyLeave{ nullptr };
    };

    void add(ListItem* item)
    {
        AT_ASSERT(item);

        item->leave();
        item->m_belongedList = this;

        m_list.push_back(item);
    }

    bool remove(ListItem* item)
    {
        AT_ASSERT(item);

        if (item->m_belongedList != this) {
            AT_ASSERT(false);
            return false;
        }

        auto it = std::find(m_list.begin(), m_list.end(), item);

        if (it != m_list.end()) {
            m_list.erase(it);

            // NOTE
            // Disable in cuda to avoid nvcc error.
            // If below code is enabled, nvcc occurs an error, but I don't know why.
#ifndef __AT_CUDA__
            for each (auto& item in m_list)
            {
                item->doFuncWhenAnyLeave();
            }

            item->m_belongedList = nullptr;
#endif

            return true;
        }

        return false;
    }

    int currentIndex(const ListItem* item)
    {
        AT_ASSERT(item);

        if (item->m_belongedList != this) {
            AT_ASSERT(false);
            return -1;
        }

        auto it = std::find(m_list.begin(), m_list.end(), item);

        if (it != m_list.end()) {
            auto ret = std::distance(m_list.begin(), it);
            return static_cast<int>(ret);
        }

        return -1;
    }

    void deleteAllDataAndClear()
    {
        auto it = m_list.begin();

        while (it != m_list.end()) {
            auto* item = *it;

            it = m_list.erase(it);

            item->m_belongedList = nullptr;

            auto data = item->getData();
            delete data;
        }

        m_list.clear();
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
